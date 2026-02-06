"""
Whisper Transcription Worker

Transcribes audio files using OpenAI Whisper.
Keeps model loaded in memory for faster subsequent transcriptions.
"""

from typing import Optional, Callable

# Global whisper model cache
_whisper_model = None
_whisper_model_name = None


def load_whisper_model(model_name: str = "medium", progress_callback: Optional[Callable[[str], None]] = None):
    """Load whisper model into memory cache"""
    global _whisper_model, _whisper_model_name

    if _whisper_model is not None and _whisper_model_name == model_name:
        if progress_callback:
            progress_callback("Whisper model already loaded")
        return

    if progress_callback:
        progress_callback(f"Loading Whisper {model_name} model...")

    import whisper
    _whisper_model = whisper.load_model(model_name)
    _whisper_model_name = model_name

    if progress_callback:
        progress_callback("Whisper model loaded!")


def unload_whisper_model():
    """Unload whisper model from memory"""
    global _whisper_model, _whisper_model_name
    _whisper_model = None
    _whisper_model_name = None

    import torch
    torch.cuda.empty_cache()


def is_whisper_loaded() -> bool:
    """Check if whisper model is loaded"""
    return _whisper_model is not None


def get_whisper_model_name() -> Optional[str]:
    """Get the name of the loaded whisper model"""
    return _whisper_model_name


def transcribe(audio_path: str, model_name: str = "medium") -> str:
    """
    Transcribe audio file using whisper.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (tiny, base, small, medium, large)

    Returns:
        Transcribed text
    """
    global _whisper_model, _whisper_model_name

    # Load model if not cached or different model requested
    if _whisper_model is None or _whisper_model_name != model_name:
        load_whisper_model(model_name)

    result = _whisper_model.transcribe(audio_path)
    return result["text"]


# PyQt6 thread wrapper
try:
    from PyQt6.QtCore import QThread, pyqtSignal

    class TranscribeThread(QThread):
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, audio_path: str, model_name: str = "medium"):
            super().__init__()
            self.audio_path = audio_path
            self.model_name = model_name

        def run(self):
            try:
                text = transcribe(self.audio_path, self.model_name)
                self.finished.emit(text)
            except Exception as e:
                self.error.emit(str(e))

except ImportError:
    # PyQt6 not available
    TranscribeThread = None
