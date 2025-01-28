package gemini_asr

import (
	"context"
	"errors"
	"go.uber.org/zap"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"
	"time"
)

type GeminiASR struct {
	*zap.Logger
	ctx               context.Context
	model             string
	readerChan        chan OutMessage
	client            *genai.Client
	generateConfig    *genai.GenerateContentConfig
	session           *genai.Session
	currentTranscript string
}

type GenerateOpts struct {
	Project  string
	Location string
	Model    string
	Cred     *google.Credentials
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
	gemini.client, err = genai.NewClient(ctx, &genai.ClientConfig{
		Project:     opts.Project,
		Location:    opts.Location,
		Backend:     genai.BackendGoogleAI,
		Credentials: opts.Cred,
	})
	if opts.Config == nil {
		opts.Config = &genai.GenerateContentConfig{}
	}
	if err != nil {
		return nil, err
	}
	gemini.generateConfig = opts.Config
	gemini.readerChan = make(chan OutMessage, 10000)
	go gemini.asyncReader()
	return gemini, err
}

func (g *GeminiASR) Read(ctx context.Context) (OutMessage, error) {
	for {
		select {
		case <-ctx.Done():
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
				subLogger.Debug("Received setup ServerContent")
				if msg.ServerContent.ModelTurn != nil {
					for _, p := range msg.ServerContent.ModelTurn.Parts {
						switch {

						case p.Text != "":
							subLogger.Debug("Received text response", zap.String("text", p.Text))
							g.currentTranscript += p.Text
							g.readerChan <- OutMessage{Text: p.Text, Final: false}

						default:
							subLogger.Debug("Received non-text", zap.Any("result", p))
						}
					}

					if msg.ServerContent.Interrupted {
						subLogger.Debug("Received interruption signal")
					}

					if msg.ServerContent.TurnComplete {
						subLogger.Debug("Received TurnComplete signal")
						g.readerChan <- OutMessage{Text: g.currentTranscript, Final: true}
						g.currentTranscript = ""
					}
				}
			}
		}
	}
}

func (g *GeminiASR) Write(bts []byte) error {
	var msg genai.LiveClientMessage
	msg.RealtimeInput = &genai.LiveClientRealtimeInput{
		MediaChunks: []*genai.Blob{{
			Data:     bts,
			MIMEType: "audio/wav",
		}},
	}
	err := g.session.Send(&msg)
	if err != nil {
		g.Error("Error in realtime input", zap.Error(err))
		return err
	}
	return nil
}
