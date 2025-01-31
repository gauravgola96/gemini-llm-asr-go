package gemini_asr

import (
	"context"
	"errors"
	"github.com/gordonklaus/portaudio"
	"go.uber.org/zap"
	"os"
	"sync"
)

func StartSession(sampleRate int) {
	logger := zap.Must(zap.NewDevelopment())
	subLogger := logger.With()

	var (
		inStream  *portaudio.Stream
		frameSize int
	)

	switch sampleRate {
	case 8000:
		frameSize = 160
		//UpSample = true
	case 16000:
		frameSize = 320
	default:
		subLogger.Error("Unsupported sample rate")
		return
	}

	inputFrame := make([]int16, frameSize)
	err := portaudio.Initialize()
	if err != nil {
		subLogger.Error("Error in portaudio initialize", zap.Error(err))
		return
	}
	ctx, canFnc := context.WithCancel(context.TODO())
	defer func() {
		canFnc()
		portaudio.Terminate()
	}()
	inStream, err = portaudio.OpenDefaultStream(1, 0, float64(sampleRate), len(inputFrame), inputFrame)
	if err != nil {
		subLogger.Error("Error portaudio stream", zap.Error(err))
		return
	}

	//SET "API_KEY" ENV : API_KEY for gemini 2.0 flash
	asr, err := NewGeminiASR(ctx, GenerateOpts{
		APIKey: os.Getenv("API_KEY"),
	})
	if err != nil {
		subLogger.Error("Error creating gemini asr", zap.Error(err))
		return
	}

	wg := &sync.WaitGroup{}
	wg.Add(1)

	go func() {
		defer wg.Done()
		for {
			message, err := asr.Read()
			if err != nil {
				return
			}
			if message.Text != "" && message.Final {
				subLogger.Info("Received out Message", zap.String("text", message.Text),
					zap.Bool("final", message.Final),
					zap.Float64("start_time", message.StartTime),
					zap.Float64("end_time", message.EndTime))
			}
		}
	}()

	err = inStream.Start()
	if err != nil {
		subLogger.Error("Error starting stream", zap.Error(err))
		return
	}
	defer inStream.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			//var buffer []byte
			err = inStream.Read()
			if errors.Is(err, portaudio.InputOverflowed) {
				continue
			}
			if err != nil {
				subLogger.Error("Error reading audio", zap.Error(err))
				return
			}

			err = asr.Write(inputFrame, sampleRate)
			if err != nil {
				return
			}
		}
	}
}
