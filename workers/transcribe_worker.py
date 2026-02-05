from PyQt6.QtCore import QThread, pyqtSignal


class TranscribeThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            import whisper
            model = whisper.load_model("medium")
            result = model.transcribe(self.audio_path)
            self.finished.emit(result["text"])
        except Exception as e:
            self.error.emit(str(e))
