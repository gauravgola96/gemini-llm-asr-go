
## Installation

Install golang 1.18+ - https://go.dev/doc/install

onnxruntime - https://onnxruntime.ai/docs/install

Macos onnxruntime - ```brew install onnxruntime```

### ENV
```bash
export LIBRARY_PATH=/opt/homebrew/Cellar/onnxruntime/1.20.1/lib

export C_INCLUDE_PATH=/opt/homebrew/Cellar/onnxruntime/1.20.1/include/onnxruntime
```

### RUN
```bash
  go mod tidy
  go run main.go
```


### Blog:
https://medium.com/@gauravgola/from-speech-to-text-building-genai-based-realtime-asr-application-with-gemini-2-0-golang-e728e45c6e5c