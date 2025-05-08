#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "typer",
#     "tiktoken",
#     "openai",
#     "tqdm",
# ]
# ///

import re
import subprocess
import tempfile
from pathlib import Path

environment_imports = (
    "ffmpeg >= 4.2",
)

import typer
import tiktoken
from tqdm import tqdm
from openai import OpenAI

app = typer.Typer(help="Split audio file on silences, transcribe via OpenAI STT, aggregate transcript, and report cost.")

# Pricing definitions: input (per minute) and output (per 1M tokens)
MODEL_PRICING = {
    "gpt-4o-transcribe":      {"unit_in": "minutes", "price_in": 0.006, "unit_out": "tokens", "price_out": 10.00},
    "gpt-4o-mini-transcribe": {"unit_in": "minutes", "price_in": 0.003, "unit_out": "tokens", "price_out": 5.00},
}

# API limit: maximum audio duration per request (seconds)
API_MAX_DURATION = 1500
# Margins
SIZE_MARGIN_BYTES = 2 * 1024 * 1024    # 2 MB
TIME_MARGIN = 100                      # 100 seconds
# Silence detection threshold
SILENCE_DB = -35       # dBFS
SILENCE_DUR = 0.5      # seconds


def detect_silences(audio: Path) -> list[float]:
    """
    Run ffmpeg silencedetect to find silence_start times in the audio.
    Returns list of silence timestamps (in seconds).
    """
    cmd = [
        "ffmpeg", "-i", str(audio),
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_DUR}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # parse lines like [silencedetect @ ...] silence_start: 12.345
    times = []
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start: (\d+\.?\d*)", line)
        if m:
            times.append(float(m.group(1)))
    return sorted(times)


def split_audio(input_file: Path, max_bytes: int, tmpdir: Path) -> list[Path]:
    """
    Split input_file into chunks not exceeding (max_bytes - SIZE_MARGIN_BYTES)
    and (API_MAX_DURATION - TIME_MARGIN) by cutting at nearest silence.
    Returns list of chunk paths.
    """
    # compute effective limits
    effective_bytes = max_bytes - SIZE_MARGIN_BYTES
    bit_rate = get_bitrate(input_file)
    duration_by_size = effective_bytes * 8 / bit_rate
    max_duration = API_MAX_DURATION - TIME_MARGIN
    target_duration = min(duration_by_size, max_duration)

    # detect silences
    silences = detect_silences(input_file)
    # include start and end
    total_duration = get_duration(input_file)
    breakpoints = [0.0] + [t for t in silences if t < total_duration] + [total_duration]

    # build segments: accumulate until close to target_duration, then cut at last silence
    chunks = []
    start = 0.0
    for bp in breakpoints[1:]:
        if bp - start >= target_duration:
            # find last silence before start+target
            cut = max([t for t in silences if start < t <= start + target_duration], default=start + target_duration)
            end = cut
            chunks.append((start, end))
            start = end
    # final segment
    if start < total_duration:
        chunks.append((start, total_duration))

    # write segments
    paths = []
    for idx, (s, e) in enumerate(chunks, 1):
        out = tmpdir / f"seg_{idx:03d}{input_file.suffix}"
        duration = e - s
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(input_file),
            "-ss", str(s),
            "-t", str(duration),
            str(out)
        ], check=True)
        paths.append(out)
    return paths


def get_bitrate(path: Path) -> float:
    """Return audio bitrate in bits per second."""
    res = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def get_duration(path: Path) -> float:
    """Return total duration of audio in seconds."""
    res = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ], capture_output=True, text=True, check=True)
    return float(res.stdout.strip())

@app.command()
def stt(
    input_file: Path = typer.Argument(..., exists=True, help="Input audio file (mp3, wav, etc.)"),
    max_size_mb: int = typer.Option(25, "-s", "--max-size-mb", help="Max chunk size in MB (default 25)"),
    stt_model: str = typer.Option("gpt-4o-transcribe", "-m", "--model", help="STT model to use"),
    output: Path = typer.Option(Path("transcript.txt"), "-o", "--output", help="Transcript output filename"),
):
    """
    1) Split INPUT_FILE on silences with size/time margins,
    2) Transcribe each with OpenAI STT → text,
    3) Concatenate transcripts,
    4) Report duration, tokens, and cost,
    5) Clean up intermediates.
    """
    client = OpenAI()
    raw_bytes = max_size_mb * 1024 * 1024
    typer.echo(f"Splitting {input_file.name} on silences (≤{max_size_mb}MB-2MB & ≤{API_MAX_DURATION}s-100s)...")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        chunks = split_audio(input_file, raw_bytes, tmpdir)
        typer.echo(f"Generated {len(chunks)} chunks via silence-based splitting.")

        transcripts, total_dur = [], 0.0
        typer.echo("🔊 Transcribing chunks:")
        for idx, chunk in enumerate(tqdm(chunks, desc=" STT chunks", unit="chunk"), start=1):
            dur = get_duration(chunk)
            total_dur += dur
            with open(chunk, "rb") as f:
                text = client.audio.transcriptions.create(
                    model=stt_model,
                    file=f,
                    response_format="text",
                )
            transcripts.append(text)

    full_text = "\n\n".join(transcripts)
    output.write_text(full_text, encoding="utf-8")

    pricing = MODEL_PRICING.get(stt_model)
    enc = tiktoken.encoding_for_model(stt_model) if pricing and pricing.get("unit_out") == "tokens" else None
    total_tokens = len(enc.encode(full_text)) if enc else None

    cost_in = (total_dur/60) * pricing["price_in"] if pricing else None
    cost_out = (total_tokens/1_000_000) * pricing["price_out"] if pricing and total_tokens is not None else None
    total_cost = ((cost_in or 0) + (cost_out or 0)) if pricing else None

    typer.echo("\n🎉 Done!")
    typer.echo(f"• Chunks processed: {len(chunks)}")
    typer.echo(f"• Total duration: {total_dur:.1f}s ({total_dur/60:.2f}m)")
    if total_tokens is not None:
        typer.echo(f"• Tokens output: {total_tokens}")
    if cost_in is not None:
        typer.echo(f"• Cost (audio @{pricing['price_in']}/min): ${cost_in:.6f}")
    if cost_out is not None:
        typer.echo(f"• Cost (text @{pricing['price_out']}/1M tokens): ${cost_out:.6f}")
    if total_cost is not None:
        typer.echo(f"• Total cost: ${total_cost:.6f}")

if __name__ == "__main__":
    app()

