import typer
from speak.speak import say
from speak.listen import transcribe

app = typer.Typer(help="Speak CLI: Audio synthesis and transcription tools")

# Add commands directly to the main app
app.command(name="say", help="Text-to-speech functionality")(say)
app.command(name="listen", help="Speech-to-text functionality")(transcribe)


def main() -> None:
    """Entry point for the CLI"""
    app()
