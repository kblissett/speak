import re
import subprocess
import tempfile
from pathlib import Path

import tiktoken
import typer
from openai import OpenAI
from tqdm import tqdm

# Pricing definitions (unit, price per 1M)
MODEL_PRICING = {
    "gpt-4o-mini-tts": {"unit": "tokens", "price": 12.00},
    "tts-1": {"unit": "characters", "price": 15.00},
    "tts-1-hd": {"unit": "characters", "price": 30.00},
}


def load_paragraphs(text: str) -> list[str]:
    return re.split(r"\n\s*\n", text.strip())


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


def say(
    input_file: Path = typer.Argument(
        None, exists=False, help="Input UTF-8 .txt file or '-' for stdin"
    ),
    max_tokens: int = typer.Option(
        1500, "-m", "--max-tokens", help="Max tokens per TTS request"
    ),
    tts_model: str = typer.Option(
        "gpt-4o-mini-tts", "--model", help="TTS model to use"
    ),
    voice: str = typer.Option("ash", "--voice", help="Built-in voice name"),
    output: Path = typer.Option(
        Path("output.mp3"), "-o", "--output", help="Final MP3 filename"
    ),
):
    """
    1) Read INPUT_FILE or stdin, split into <=MAX_TOKENS chunks on paragraphs,
    2) Stream each to OpenAI TTS → WAV,
    3) Concatenate via ffmpeg concat demuxer,
    4) Transcode final WAV to MP3 via ffmpeg,
    5) Report tokens/characters, duration, cost,
    6) Clean up all intermediate files.
    """
    # Read from stdin if input_file is None or '-'
    if input_file is None or str(input_file) == "-":
        import sys

        raw = sys.stdin.read()
    else:
        # Verify file exists when not using stdin
        if not input_file.exists():
            typer.echo(f"Error: Input file '{input_file}' does not exist.", err=True)
            raise typer.Exit(1)
        raw = input_file.read_text(encoding="utf-8")
    total_chars = len(raw)
    enc = tiktoken.encoding_for_model(tts_model)
    paras = load_paragraphs(raw)
    total_tokens = sum(count_tokens(p, enc) for p in paras)

    typer.echo(
        f"Loaded {len(paras)} paragraphs → {total_tokens} tokens, {total_chars} chars"
    )

    chunks = chunk_paragraphs(paras, max_tokens, enc)
    typer.echo(f"Split into {len(chunks)} chunks (≤{max_tokens} tokens)")

    client = OpenAI()

    # Create a single tempdir for all intermediates
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        wav_parts: list[Path] = []
        typer.echo("📣 Synthesizing chunks:")
        for idx, chunk in enumerate(
            tqdm(chunks, desc=" TTS chunks", unit="chunk"), start=1
        ):
            text = "\n\n".join(chunk)
            part = tmpdir / f"chunk_{idx:03}.wav"
            with client.audio.speech.with_streaming_response.create(
                model=tts_model,
                voice=voice,
                input=text,
                response_format="wav",
            ) as resp:
                resp.stream_to_file(part)
            wav_parts.append(part)

        # Use ffmpeg concat demuxer to join wavs and transcode to mp3 in one step
        concat_list = tmpdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in wav_parts),
            encoding="utf-8",
        )

        typer.echo("🔗 Concatenating and transcoding to MP3 via ffmpeg…")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(output),
            ],
            check=True,
        )

        # Done with tmpdir; it and all chunk_*.wav + concat.txt get auto-deleted here

    # cost calc
    pricing = MODEL_PRICING[tts_model]
    if pricing["unit"] == "tokens":
        cost = total_tokens / 1_000_000 * pricing["price"]
        unit = "tokens"
    else:
        cost = total_chars / 1_000_000 * pricing["price"]
        unit = "characters"

    typer.echo("\n🎉 Done!")
    typer.echo(f"• Tokens processed: {total_tokens}")
    typer.echo(f"• Characters processed: {total_chars}")
    if cost is not None:
        typer.echo(f"• Exact cost (@ ${pricing['price']}/1M {unit}): ${cost:.6f}")


if __name__ == "__main__":
    # Create a simple Typer app for direct execution
    cli = typer.Typer()
    cli.command()(say)
    cli()
