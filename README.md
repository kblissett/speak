# speak

_Built with the help of **o4-mini-high**._

A simple, one-file, CLI tool for accessing OpenAI TTS models.

## Installation

No virtualenv or `pip install` needed—just make sure you have `uv` and `ffmpeg` on your `PATH`.

```bash
# Clone this repo
git clone https://github.com/kblissett/speak.git
cd speak

chmod +x speak.py
./speak.py $INPUT_FILE -o $OUTPUT_FILE
```

## Requirements

- **uv** (https://github.com/uv/uv) — for the “shebang” auto-venv runner  
- **ffmpeg** (with `ffprobe`) — for MP3 concat and duration measurement  

