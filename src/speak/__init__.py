import typer
from speak.speak import app as speak_app
from speak.listen import app as listen_app

app = typer.Typer(help="Speak CLI: Audio synthesis and transcription tools")

# Add subcommands
app.add_typer(speak_app, name="tts", help="Text-to-speech functionality")
app.add_typer(listen_app, name="stt", help="Speech-to-text functionality")


def main() -> None:
    """Entry point for the CLI"""
    app()
