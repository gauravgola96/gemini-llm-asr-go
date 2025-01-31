package main

import (
	gemini_asr "github.com/gauravgola96/gemini-llm-asr-go/gemini-asr"
	"go.uber.org/zap"
)

func init() {
	zap.ReplaceGlobals(zap.Must(zap.NewProduction()))
}

func main() {
	SAMPLERATE := 16000
	//SET "API_KEY" ENV : API_KEY for gemini 2.0 flash
	gemini_asr.StartSession(SAMPLERATE)
}
