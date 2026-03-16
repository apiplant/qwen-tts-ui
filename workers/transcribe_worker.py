"""
Parakeet Transcription Worker

Transcribes audio files using NVIDIA NeMo Parakeet v3.
Keeps model loaded in memory for faster subsequent transcriptions.
"""

from typing import Optional, Callable

# Global model cache
_asr_model = None
_asr_model_name = None


def load_parakeet_model(model_name: str = "nvidia/parakeet-tdt-0.6b-v3", progress_callback: Optional[Callable[[str], None]] = None):
    """Load Parakeet ASR model into memory cache"""
    global _asr_model, _asr_model_name

    if _asr_model is not None and _asr_model_name == model_name:
        if progress_callback:
            progress_callback("Parakeet model already loaded")
        return

    if progress_callback:
        progress_callback(f"Loading Parakeet model ({model_name})...")

    import nemo.collections.asr as nemo_asr
    _asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    _asr_model_name = model_name

    if progress_callback:
        progress_callback("Parakeet model loaded!")


def unload_parakeet_model():
    """Unload Parakeet model from memory"""
    global _asr_model, _asr_model_name
    _asr_model = None
    _asr_model_name = None

    import torch
    torch.cuda.empty_cache()


def is_parakeet_loaded() -> bool:
    """Check if Parakeet model is loaded"""
    return _asr_model is not None


def get_parakeet_model_name() -> Optional[str]:
    """Get the name of the loaded Parakeet model"""
    return _asr_model_name


def transcribe(audio_path: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v3") -> str:
    """
    Transcribe audio file using NeMo Parakeet.

    Args:
        audio_path: Path to audio file
        model_name: NeMo ASR model name

    Returns:
        Transcribed text
    """
    global _asr_model, _asr_model_name

    # Load model if not cached or different model requested
    if _asr_model is None or _asr_model_name != model_name:
        load_parakeet_model(model_name)

    transcriptions = _asr_model.transcribe([audio_path])
    return transcriptions[0] if isinstance(transcriptions[0], str) else transcriptions[0].text


# PyQt6 thread wrapper
try:
    from PyQt6.QtCore import QThread, pyqtSignal

    class TranscribeThread(QThread):
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, audio_path: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
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
