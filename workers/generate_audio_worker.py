import os
from datetime import datetime
from pathlib import Path
import soundfile as sf
from PyQt6.QtCore import QThread, pyqtSignal


class GenerateAudioThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model, text, language, ref_audio=None, ref_text=None, instruct=None):
        super().__init__()
        self.model = model
        self.text = text
        self.language = language
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.instruct = instruct
    
    def run(self):
        try:
            kwargs = {
                'text': self.text,
                'language': self.language,
            }
            
            if self.ref_audio and self.ref_text:
                kwargs['ref_audio'] = self.ref_audio
                kwargs['ref_text'] = self.ref_text
            
            if self.instruct:
                kwargs['instruct'] = [self.instruct]
            
            wavs, sr = self.model.generate_voice_clone(**kwargs)
            
            # Generate output filename
            if self.ref_audio:
                voice_name = Path(self.ref_audio).stem
            else:
                voice_name = "no_reference"
            timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
            output_path = f"output_voices/{voice_name}_{timestamp}.wav"
            
            os.makedirs("output_voices", exist_ok=True)
            sf.write(output_path, wavs[0], sr)
            
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))
