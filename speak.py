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

import typer
import tiktoken
from tqdm import tqdm
from openai import OpenAI

app = typer.Typer(help="Split text, synthesize via OpenAI TTS, concat, and clean up intermediates.")

# Pricing definitions (unit, price per 1M)
MODEL_PRICING = {
    "gpt-4o-mini-tts":       {"unit": "tokens",     "price": 12.00},
    "tts-1":                 {"unit": "characters", "price": 15.00},
    "tts-1-hd":              {"unit": "characters", "price": 30.00},
}

def load_paragraphs(text: str) -> list[str]:
    return re.split(r'\n\s*\n', text.strip())

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))

def chunk_paragraphs(paragraphs: list[str], max_tokens: int, enc) -> list[list[str]]:
    chunks: list[list[str]] = []
    current, current_tokens = [], 0
    for p in paragraphs:
        p_tokens = count_tokens(p, enc)
        if p_tokens > max_tokens:
            # flush current
            if current:
                chunks.append(current)
                current, current_tokens = [], 0
            # split large paragraph
            token_ids = enc.encode(p)
            for i in range(0, len(token_ids), max_tokens):
                sub_text = enc.decode(token_ids[i : i + max_tokens])
                chunks.append([sub_text])
            continue
        # rollover if needed
        if current and current_tokens + p_tokens > max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(p)
        current_tokens += p_tokens
    if current:
        chunks.append(current)
    return chunks

@app.command()
def tts(
    input_file: Path = typer.Argument(..., exists=True, help="Input UTF-8 .txt file"),
    max_tokens: int = typer.Option(1500, "-m", "--max-tokens", help="Max tokens per TTS request"),
    tts_model: str = typer.Option("gpt-4o-mini-tts", "--model", help="TTS model to use"),
    voice: str = typer.Option("coral", "--voice", help="Built-in voice name"),
    output: Path = typer.Option(Path("output.mp3"), "-o", "--output", help="Final MP3 filename"),
):
    """
    1) Read INPUT_FILE, split into <=MAX_TOKENS chunks on paragraphs,
    2) Stream each to OpenAI TTS → MP3,
    3) Concatenate via ffmpeg,
    4) Report tokens/characters, duration, cost,
    5) Clean up all intermediate files.
    """
    raw = input_file.read_text(encoding="utf-8")
    total_chars = len(raw)
    enc = tiktoken.encoding_for_model(tts_model)
    paras = load_paragraphs(raw)
    total_tokens = sum(count_tokens(p, enc) for p in paras)

    typer.echo(f"Loaded {len(paras)} paragraphs → {total_tokens} tokens, {total_chars} chars")

    chunks = chunk_paragraphs(paras, max_tokens, enc)
    typer.echo(f"Split into {len(chunks)} chunks (≤{max_tokens} tokens)")

    client = OpenAI()

    # Create a single tempdir for all intermediates
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        mp3_parts: list[Path] = []
        typer.echo("📣 Synthesizing chunks:")
        for idx, chunk in enumerate(tqdm(chunks, desc=" TTS chunks", unit="chunk"), start=1):
            text = "\n\n".join(chunk)
            part = tmpdir / f"chunk_{idx:03}.mp3"
            with client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=voice,
                input=text,
                response_format="mp3",
            ) as resp:
                resp.stream_to_file(part)
            mp3_parts.append(part)

        # write ffmpeg concat list inside tmpdir
        concat_list = tmpdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in mp3_parts),
            encoding="utf-8",
        )

        typer.echo("🔗 Concatenating via ffmpeg…")
        subprocess.run([
            "ffmpeg",
            "-hide_banner", "-loglevel", "error", "-nostats",
            "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            str(output),
        ], check=True)

        # Done with tmpdir; it and all chunk_*.mp3 + concat.txt get auto-deleted here

    # measure duration
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(output),
    ], capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())

    # cost calc
    pricing = MODEL_PRICING.get(tts_model)
    if pricing:
        if pricing["unit"] == "tokens":
            cost = total_tokens / 1_000_000 * pricing["price"]
            unit = "tokens"
        else:
            cost = total_chars / 1_000_000 * pricing["price"]
            unit = "characters"
    else:
        cost = None

    typer.echo("\n🎉 Done!")
    typer.echo(f"• Tokens processed: {total_tokens}")
    typer.echo(f"• Characters processed: {total_chars}")
    typer.echo(f"• Audio duration: {duration:.1f}s ({duration/60:.2f}m)")
    if cost is not None:
        typer.echo(f"• Exact cost (@ ${pricing['price']}/1M {unit}): ${cost:.6f}")

if __name__ == "__main__":
    app()
