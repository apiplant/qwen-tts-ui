import os
from datetime import datetime
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal


class GenerateVideoThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, avatar_path, audio_path, padding_before, padding_after):
        super().__init__()
        self.avatar_path = avatar_path
        self.audio_path = audio_path
        self.padding_before = padding_before
        self.padding_after = padding_after

    def run(self):
        try:
            from moviepy import ImageClip, AudioFileClip, concatenate_audioclips
            import numpy as np
            
            # Load audio
            audio = AudioFileClip(self.audio_path)
            
            # Create silence clips for padding using numpy arrays
            audio_clips = []
            
            if self.padding_before > 0:
                # Create silence array matching audio properties
                silence_before = np.zeros((int(self.padding_before * audio.fps), audio.nchannels if hasattr(audio, 'nchannels') else 2))
                from moviepy import AudioArrayClip
                silence_before_clip = AudioArrayClip(silence_before, fps=audio.fps)
                audio_clips.append(silence_before_clip)
            
            audio_clips.append(audio)
            
            if self.padding_after > 0:
                # Create silence array matching audio properties
                silence_after = np.zeros((int(self.padding_after * audio.fps), audio.nchannels if hasattr(audio, 'nchannels') else 2))
                from moviepy import AudioArrayClip
                silence_after_clip = AudioArrayClip(silence_after, fps=audio.fps)
                audio_clips.append(silence_after_clip)
            
            # Concatenate audio with padding
            final_audio = concatenate_audioclips(audio_clips)
            
            # Create video from image
            video = ImageClip(self.avatar_path, duration=final_audio.duration)
            video = video.with_audio(final_audio)
            
            # Generate output filename
            avatar_name = Path(self.avatar_path).stem
            audio_name = Path(self.audio_path).stem
            timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
            output_path = f"output_avatars/{avatar_name}_{audio_name}_{timestamp}.mp4"
            
            os.makedirs("output_avatars", exist_ok=True)
            video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24, logger=None)
            
            # Clean up
            video.close()
            audio.close()
            final_audio.close()
            
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))
