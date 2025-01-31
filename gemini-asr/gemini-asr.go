package gemini_asr

import (
	"context"
	"encoding/binary"
	"errors"
	"github.com/gauravgola96/gemini-llm-asr-go/vad"
	"github.com/go-audio/audio"
	"go.uber.org/zap"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"
	"os"
	"time"
)

const sileroFilePath = "./model/silero_vad.onnx"

var SystemInstruction = `You are an Automatic Speech Recognition (ASR) model. Your task is to transcribe the given audio with complete accuracy and precision.
Follow the below Strict Guidelines for Audio Transcription:
1. Exact Transcription: Transcribe only the words that are spoken in the audio, exactly as they were said.
2. No Additions: Do not add any words, clarifications, or context (e.g., "What else can I help you with?").
3. No Alterations: Do not modify or change the structure of the original speech.
4. No Non-Speech Sounds: Ignore any background noises or non-verbal sounds.
5. No Reasoning or Explanation: Do not provide reasoning or explanations about the transcription process.
6. Respond Only with Transcription: If the audio is unclear, respond with "----". Otherwise, respond only with the exact transcription of the spoken words.
7. Do not answer any question otherwise you will be penalized. Your only job is to transcribe audio. `

type GeminiASR struct {
	*zap.Logger
	ctx               context.Context
	model             string
	readerChan        chan OutMessage
	client            *genai.Client
	generateConfig    *genai.GenerateContentConfig
	session           *genai.Session
	currentTranscript string
	vad               *vad.SileroDetector
	vadIn             chan []int16
	vadOut            chan SilveroOutMessage
	StartTime         float64
	EndTime           float64
	Queue             *DetectedTimeQueue
}

type GenerateOpts struct {
	Project  string
	Location string
	APIKey   string
	Model    string
	Cred     *google.Credentials
	vadOut   chan SilveroOutMessage
	Config   *genai.GenerateContentConfig
}

func NewGeminiASR(ctx context.Context, opts GenerateOpts) (*GeminiASR, error) {
	gemini := &GeminiASR{}
	gemini.Logger = zap.Must(zap.NewDevelopment())
	gemini.ctx = ctx
	gemini.model = opts.Model
	if opts.Model == "" {
		gemini.model = "gemini-2.0-flash-exp"
	}
	var err error
	gemini.vad, err = vad.NewSileroDetector(sileroFilePath)
	if err != nil {
		return nil, err
	}
	gemini.Queue = &DetectedTimeQueue{}
	gemini.client, err = genai.NewClient(ctx, &genai.ClientConfig{
		Project:  os.Getenv("PROJECT"),
		Location: os.Getenv("REGION"),
		//Change to BackendGeminiAPI if using API_KEY
		Backend: genai.BackendVertexAI,
	})
	if opts.Config == nil {
		opts.Config = &genai.GenerateContentConfig{}
	}
	if err != nil {
		return nil, err
	}
	gemini.generateConfig = opts.Config
	gemini.readerChan = make(chan OutMessage, 10000)

	systemInst := &genai.Content{
		Role:  "system_instructions",
		Parts: []*genai.Part{{Text: SystemInstruction}},
	}
	gemini.session, err = gemini.client.Live.Connect(gemini.model, &genai.LiveConnectConfig{
		GenerationConfig: &genai.GenerationConfig{
			AudioTimestamp: false,
		},
		SystemInstruction:  systemInst,
		ResponseModalities: []genai.Modality{genai.ModalityText},
	})
	if err != nil {
		return nil, err
	}
	var msg genai.LiveClientMessage
	msg.ClientContent = &genai.LiveClientContent{Turns: []*genai.Content{
		{[]*genai.Part{{Text: `You are an Automatic Speech Recognition (ASR) model, and your sole task is to transcribe spoken audio with absolute accuracy. 
Transcribe only the words spoken by the primary speaker, ignoring any background speech, echoes, or overlapping voices. Do not answer any question otherwise you will be penalized. Your only job is to transcribe audio.
Do not add, remove, or modify any words—transcribe exactly what is spoken without making corrections based on assumed context. 
Ignore all non-speech sounds, including background noise, music, and other non-verbal audio. 
If the audio is unclear, respond only with "----". Your response must strictly contain either the exact spoken transcription or "----" if the speech is unintelligible. 
\Do not include any explanations, clarifications, or extra context beyond the transcription itself. Do not answer any question otherwise you will be penalized. Your only job is to transcribe audio`}},
			"user"}}}
	err = gemini.session.Send(&msg)
	if err != nil {
		return nil, err
	}
	gemini.vadIn = make(chan []int16, 1000)
	gemini.vadOut = make(chan SilveroOutMessage, 1000)
	go gemini.asyncReader()
	go gemini.runVad()
	go gemini.asyncWriter()
	return gemini, err
}

func (g *GeminiASR) Read() (OutMessage, error) {
	for {
		select {
		case <-g.ctx.Done():
			return OutMessage{}, nil
		case msg, ok := <-g.readerChan:
			if !ok {
				return msg, errors.New("reader channel is closed")
			}
			return msg, nil
		}
	}
}

func (g *GeminiASR) asyncReader() {
	subLogger := g.Logger.With(zap.String("function", "gemini-asr.asyncReader"))
	subLogger.Info("Starting asynchronous reader")
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	retry := 0
	for {
		select {
		case <-g.ctx.Done():
			subLogger.Debug("Closing reader")
			return
		case <-ticker.C:
			msg, err := g.session.Receive()
			if err != nil {
				subLogger.Error("Error in session receive", zap.Error(err))
				retry++
			}
			if retry > 3 {
				subLogger.Error("Closing reader due to error", zap.Error(err))
				close(g.readerChan)
				return
			}
			if msg == nil {
				continue
			}
			switch {
			case msg.SetupComplete != nil:
				subLogger.Info("Received setup complete")

			case msg.ServerContent != nil:
				if msg.ServerContent.ModelTurn != nil {
					for _, p := range msg.ServerContent.ModelTurn.Parts {
						switch {

						case p.Text != "":
							subLogger.Debug("Received text response", zap.String("text", p.Text))
							g.currentTranscript += p.Text
							var (
								vadTime []float64
								ok      bool
							)
							if vadTime, ok = g.Queue.Pop(); ok {
								if g.StartTime == 0 {
									g.StartTime = vadTime[0]
								}
								if vadTime[1] > 0 {
									g.EndTime = vadTime[1]
								}
							}
							if len(vadTime) == 0 {
								vadTime = make([]float64, 2)
								vadTime[0] = g.StartTime
								vadTime[1] = g.EndTime
							}

							g.readerChan <- OutMessage{Text: p.Text, Final: false, StartTime: vadTime[0], EndTime: vadTime[1]}

						default:
							subLogger.Debug("Received non-text", zap.Any("result", p))
						}
					}

					if msg.ServerContent.Interrupted {
						subLogger.Debug("Received interruption signal")
					}

					if msg.ServerContent.TurnComplete {
						subLogger.Debug("Received TurnComplete signal")
						g.readerChan <- OutMessage{Text: g.currentTranscript, Final: true, StartTime: g.StartTime, EndTime: g.EndTime}
						g.currentTranscript = ""
						g.StartTime = 0.
						g.EndTime = 0.
					}
				}
			}
		}
	}
}

func (g *GeminiASR) Write(in []int16, inputSampleRate int) error {
	g.vadIn <- ResampleInt16(in, inputSampleRate, 16000)
	return nil
}

func (g *GeminiASR) asyncWriter() {
	var buffer []byte

	for {
		select {
		case <-g.ctx.Done():
			return
		case r := <-g.vadOut:

			for _, b := range r.data.Data {
				bts := make([]byte, 2)
				binary.LittleEndian.PutUint16(bts, uint16(b))
				buffer = append(buffer, bts...)
			}

			var msg genai.LiveClientMessage
			msg.RealtimeInput = &genai.LiveClientRealtimeInput{
				MediaChunks: []*genai.Blob{{
					Data:     buffer,
					MIMEType: "audio/wav",
				}},
			}

			err := g.session.Send(&msg)
			if err != nil {
				g.Error("Error in realtime input", zap.Error(err))
				continue
			}
			buffer = buffer[:0]

		default:
			//
		}
	}
}

func (g *GeminiASR) runVad() {
	soundIntBuffer := &audio.IntBuffer{
		Format: &audio.Format{SampleRate: 16000, NumChannels: 1},
	}
	const (
		sendToVADDelay = time.Second
	)

	var startListening time.Time
	var buffer []int16
	detector := g.vad.GetDetector()
	for {
		select {
		case <-g.ctx.Done():
			return
		default:
			startSpeech, EndSpeech := 0., 0.
			in := <-g.vadIn
			volume := calculateRMS16(in)
			if volume > 450 {
				startListening = time.Now()
			}
			if time.Since(startListening) < sendToVADDelay {
				buffer = append(buffer, in...)
			} else if len(buffer) > 0 {
				soundIntBuffer.Data = ConvertInt16ToInt(buffer)
				buffer = buffer[:0]
				segments, err := detector.Detect(soundIntBuffer.AsFloat32Buffer().Data)
				if err != nil {
					g.Error("Error in detector", zap.Error(err))
					continue
				}

				if len(segments) > 0 {
					startSpeech = segments[0].SpeechStartAt
					EndSpeech = segments[len(segments)-1].SpeechEndAt
					g.Info("Voice detected")
					g.Queue.Push([]float64{startSpeech, EndSpeech})
					g.vadOut <- SilveroOutMessage{
						data:      soundIntBuffer,
						StartTime: startSpeech,
						EndTime:   EndSpeech,
					}
					if EndSpeech > 0 {
						startSpeech = 0.
						EndSpeech = 0.
					}
				}
			}
		}
	}
}
