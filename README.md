# speak

Audio CLI toolkit for text-to-speech and speech-to-text using OpenAI models.

## Features

- **say**: Text-to-speech (TTS) using OpenAI's TTS models.
- **listen**: Speech-to-text (STT) using OpenAI's transcription models.

## Installation

### Requirements

- Python ≥3.13
- FFmpeg (with `ffprobe`) on your `PATH`

### Install via pip

```bash
pip install speak
```

> Or install from source:

```bash
git clone https://github.com/kblissett/speak.git
cd speak
pip install .
```

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Text-to-speech (say)

```bash
speak say input.txt -o output.mp3
```

Options:

```bash
speak say \
  --model gpt-4o-mini-tts \
  --voice coral \
  --max-tokens 1500 \
  input.txt \
  -o output.mp3
```

### Speech-to-text (listen)

```bash
speak listen input.mp3 -o transcript.txt
```

Options:

```bash
speak listen \
  --model gpt-4o-transcribe \
  --max-size-mb 25 \
  input.mp3 \
  -o transcript.txt
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

