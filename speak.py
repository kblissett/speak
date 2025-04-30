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
from pathlib import Path

import typer
import tiktoken
from tqdm import tqdm
from openai import OpenAI

app = typer.Typer(help="Split text at paragraphs, synthesize via OpenAI TTS, and report cost/duration.")

# Pricing definitions keyed by model name
# unit:  "tokens"  → cost per 1M input tokens
#        "characters" → cost per 1M input characters
MODEL_PRICING = {
    "gpt-4o-mini-tts":       {"unit": "tokens",     "price": 0.60},
    "gpt-4o-audio-preview":  {"unit": "tokens",     "price": 2.50},
    "tts-1":                 {"unit": "characters", "price": 15.00},
    "tts-1-hd":              {"unit": "characters", "price": 30.00},
}

def load_paragraphs(text: str) -> list[str]:
    # Split on one-or-more blank lines
    return re.split(r'\n\s*\n', text.strip())

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))

@app.command()
def tts(
    input_file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="Input UTF-8 .txt file"),
    max_tokens: int = typer.Option(2000, "-m", "--max-tokens", help="Max tokens per TTS request"),
    tts_model: str = typer.Option("gpt-4o-mini-tts", "--model", help="TTS model to use"),
    voice: str = typer.Option("coral", "--voice", help="Built-in voice name"),
    output: Path = typer.Option(Path("output.mp3"), "-o", "--output", help="Final concatenated MP3"),
):
    """
    1) Read INPUT_FILE, split into ≤MAX_TOKENS paragraphs,
    2) Stream each chunk to the TTS endpoint (MP3),
    3) Concatenate via ffmpeg,
    4) Report total tokens/characters, duration, and exact cost.
    """
    # Read raw text (for char‐based billing)
    raw_text = input_file.read_text(encoding="utf-8")
    total_chars = len(raw_text)

    # Determine tokenizer for the model
    enc = tiktoken.encoding_for_model(tts_model)

    # Split into paragraphs
    paragraphs = load_paragraphs(raw_text)
    total_tokens = sum(count_tokens(p, enc) for p in paragraphs)
    typer.echo(f"Loaded {len(paragraphs)} paragraphs → {total_tokens} tokens, {total_chars} characters total")

    # Chunk on token limit
    chunks: list[list[str]] = []
    current, current_tokens = [], 0
    for p in paragraphs:
        pt = count_tokens(p, enc)
        if current and current_tokens + pt > max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(p)
        current_tokens += pt
    if current:
        chunks.append(current)
    typer.echo(f"Split into {len(chunks)} chunks (≤{max_tokens} tokens each)")

    client = OpenAI()
    mp3_parts: list[Path] = []

    typer.echo("📣 Synthesizing with TTS:")
    for idx, chunk in enumerate(tqdm(chunks, desc=" TTS chunks", unit="chunk"), start=1):
        text = "\n\n".join(chunk)
        part_file = Path(f"chunk_{idx:03}.mp3")
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=voice,
            input=text,
            response_format="mp3",    # ensure MP3 output
        ) as resp:
            resp.stream_to_file(part_file)
        mp3_parts.append(part_file)

    # Build ffmpeg concat list
    concat_list = Path("concat.txt")
    concat_list.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in mp3_parts),
        encoding="utf-8",
    )

    typer.echo("🔗 Concatenating via ffmpeg…")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy", str(output),
    ], check=True)

    # Measure duration
    ff = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(output),
    ], capture_output=True, text=True, check=True)
    duration = float(ff.stdout.strip())

    # Compute exact cost
    pricing = MODEL_PRICING.get(tts_model)
    if pricing:
        if pricing["unit"] == "tokens":
            cost = total_tokens / 1_000_000 * pricing["price"]
            unit_desc = "tokens"
        else:
            cost = total_chars / 1_000_000 * pricing["price"]
            unit_desc = "characters"
        typer.echo("\n🎉 Done!")
        typer.echo(f"• Input processed: {total_tokens} tokens, {total_chars} characters")
        typer.echo(f"• Audio length: {duration:.1f}s ({duration/60:.2f} min)")
        typer.echo(f"• Cost for {tts_model} (@ ${pricing['price']}/1M {unit_desc}): ${cost:.6f}")
    else:
        typer.echo(f"⚠️ No pricing info for model '{tts_model}'. Skipping cost calc.")

if __name__ == "__main__":
    app()
