import sys
import os
import json
from datetime import datetime
from pathlib import Path
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox, QFileDialog,
    QProgressBar, QCheckBox, QMessageBox, QLineEdit, QListWidget,
    QListWidgetItem, QSplitter, QSlider, QStyle, QDoubleSpinBox,
    QScrollArea, QFrame
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from qwen_tts import Qwen3TTSModel
from workers import TranscribeThread, GenerateAudioThread, GenerateVideoThread
from workers import (
    load_wan_models, unload_wan_models, get_wan_model_cache,
    load_whisper_model, unload_whisper_model, is_whisper_loaded,
)


class TTSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_voice_path = None
        self.history_file = "history.json"
        self.history = self.load_history()

        # Audio player setup
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

        # Video player setup
        self.video_player = QMediaPlayer()
        self.video_audio_output = QAudioOutput()
        self.video_player.setAudioOutput(self.video_audio_output)
        self.video_player.positionChanged.connect(self.on_video_position_changed)
        self.video_player.durationChanged.connect(self.on_video_duration_changed)
        self.video_player.playbackStateChanged.connect(self.on_video_playback_state_changed)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Qwen TTS Voice Cloner")
        self.setMinimumSize(1200, 900)

        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - main controls
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)

        # Right panel - history and audio library
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # History section
        right_layout.addWidget(QLabel("History:"))
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_clicked)
        right_layout.addWidget(self.history_list)

        history_btn_layout = QHBoxLayout()
        self.delete_history_btn = QPushButton("Delete Selected")
        self.delete_history_btn.clicked.connect(self.delete_history_item)
        history_btn_layout.addWidget(self.delete_history_btn)
        self.clear_history_btn = QPushButton("Clear All")
        self.clear_history_btn.clicked.connect(self.clear_history)
        history_btn_layout.addWidget(self.clear_history_btn)
        right_layout.addLayout(history_btn_layout)

        # Audio library section
        right_layout.addWidget(QLabel("Audio Library:"))

        self.audio_library_list = QListWidget()
        self.audio_library_list.itemClicked.connect(self.on_audio_library_clicked)
        self.audio_library_list.itemDoubleClicked.connect(self.on_audio_library_double_clicked)
        right_layout.addWidget(self.audio_library_list)

        library_btn_layout = QHBoxLayout()
        self.refresh_library_btn = QPushButton("Refresh")
        self.refresh_library_btn.clicked.connect(self.refresh_audio_library)
        library_btn_layout.addWidget(self.refresh_library_btn)
        self.delete_audio_btn = QPushButton("Delete File")
        self.delete_audio_btn.clicked.connect(self.delete_audio_file)
        library_btn_layout.addWidget(self.delete_audio_btn)
        right_layout.addLayout(library_btn_layout)

        # Audio player controls (right below audio list)
        player_group = QWidget()
        player_layout = QVBoxLayout(player_group)
        player_layout.setContentsMargins(0, 5, 0, 5)

        self.now_playing_label = QLabel("No audio loaded")
        self.now_playing_label.setWordWrap(True)
        player_layout.addWidget(self.now_playing_label)

        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)

        self.time_label = QLabel("0:00 / 0:00")
        controls_layout.addWidget(self.time_label)

        player_layout.addLayout(controls_layout)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.change_volume)
        volume_layout.addWidget(self.volume_slider)
        player_layout.addLayout(volume_layout)

        right_layout.addWidget(player_group)

        # Video library section
        right_layout.addWidget(QLabel("Video Library:"))

        self.video_library_list = QListWidget()
        self.video_library_list.itemClicked.connect(self.on_video_library_clicked)
        self.video_library_list.itemDoubleClicked.connect(self.on_video_library_double_clicked)
        right_layout.addWidget(self.video_library_list)

        video_library_btn_layout = QHBoxLayout()
        self.refresh_video_library_btn = QPushButton("Refresh")
        self.refresh_video_library_btn.clicked.connect(self.refresh_video_library)
        video_library_btn_layout.addWidget(self.refresh_video_library_btn)
        self.delete_video_btn = QPushButton("Delete File")
        self.delete_video_btn.clicked.connect(self.delete_video_file)
        video_library_btn_layout.addWidget(self.delete_video_btn)
        right_layout.addLayout(video_library_btn_layout)

        # Video player controls (right below video list)
        video_player_group = QWidget()
        video_player_layout = QVBoxLayout(video_player_group)
        video_player_layout.setContentsMargins(0, 5, 0, 5)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(200)
        self.video_player.setVideoOutput(self.video_widget)
        video_player_layout.addWidget(self.video_widget)

        self.video_now_playing_label = QLabel("No video loaded")
        self.video_now_playing_label.setWordWrap(True)
        video_player_layout.addWidget(self.video_now_playing_label)

        video_controls_layout = QHBoxLayout()
        self.video_play_btn = QPushButton()
        self.video_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.video_play_btn.clicked.connect(self.toggle_video_playback)
        self.video_play_btn.setEnabled(False)
        video_controls_layout.addWidget(self.video_play_btn)

        self.video_stop_btn = QPushButton()
        self.video_stop_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.video_stop_btn.clicked.connect(self.stop_video_playback)
        self.video_stop_btn.setEnabled(False)
        video_controls_layout.addWidget(self.video_stop_btn)

        self.video_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_position_slider.sliderMoved.connect(self.set_video_position)
        video_controls_layout.addWidget(self.video_position_slider)

        self.video_time_label = QLabel("0:00 / 0:00")
        video_controls_layout.addWidget(self.video_time_label)

        video_player_layout.addLayout(video_controls_layout)

        video_volume_layout = QHBoxLayout()
        video_volume_layout.addWidget(QLabel("Volume:"))
        self.video_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_volume_slider.setRange(0, 100)
        self.video_volume_slider.setValue(100)
        self.video_volume_slider.valueChanged.connect(self.change_video_volume)
        video_volume_layout.addWidget(self.video_volume_slider)
        video_player_layout.addLayout(video_volume_layout)

        right_layout.addWidget(video_player_group)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # === Model Status Section ===
        models_group = QWidget()
        models_group.setStyleSheet("QWidget { background-color: #2a2a2a; border-radius: 5px; padding: 5px; }")
        models_layout = QHBoxLayout(models_group)
        models_layout.setContentsMargins(10, 5, 10, 5)

        # Qwen TTS model status
        qwen_layout = QVBoxLayout()
        qwen_header = QHBoxLayout()
        qwen_header.addWidget(QLabel("Qwen TTS:"))
        self.qwen_status_label = QLabel("Not Loaded")
        self.qwen_status_label.setStyleSheet("color: #ff6b6b;")
        qwen_header.addWidget(self.qwen_status_label)
        qwen_header.addStretch()
        qwen_layout.addLayout(qwen_header)
        qwen_btn_layout = QHBoxLayout()
        self.qwen_load_btn = QPushButton("Load")
        self.qwen_load_btn.setMaximumWidth(60)
        self.qwen_load_btn.clicked.connect(self.load_qwen_model)
        qwen_btn_layout.addWidget(self.qwen_load_btn)
        self.qwen_unload_btn = QPushButton("Unload")
        self.qwen_unload_btn.setMaximumWidth(60)
        self.qwen_unload_btn.clicked.connect(self.unload_qwen_model)
        self.qwen_unload_btn.setEnabled(False)
        qwen_btn_layout.addWidget(self.qwen_unload_btn)
        qwen_layout.addLayout(qwen_btn_layout)
        models_layout.addLayout(qwen_layout)

        models_layout.addWidget(QLabel("|"))

        # Whisper model status
        whisper_layout = QVBoxLayout()
        whisper_header = QHBoxLayout()
        whisper_header.addWidget(QLabel("Whisper:"))
        self.whisper_status_label = QLabel("Not Loaded")
        self.whisper_status_label.setStyleSheet("color: #ff6b6b;")
        whisper_header.addWidget(self.whisper_status_label)
        whisper_header.addStretch()
        whisper_layout.addLayout(whisper_header)
        whisper_btn_layout = QHBoxLayout()
        self.whisper_load_btn = QPushButton("Load")
        self.whisper_load_btn.setMaximumWidth(60)
        self.whisper_load_btn.clicked.connect(self.load_whisper_model)
        whisper_btn_layout.addWidget(self.whisper_load_btn)
        self.whisper_unload_btn = QPushButton("Unload")
        self.whisper_unload_btn.setMaximumWidth(60)
        self.whisper_unload_btn.clicked.connect(self.unload_whisper_model)
        self.whisper_unload_btn.setEnabled(False)
        whisper_btn_layout.addWidget(self.whisper_unload_btn)
        whisper_layout.addLayout(whisper_btn_layout)
        models_layout.addLayout(whisper_layout)

        models_layout.addWidget(QLabel("|"))

        # Wan model status
        wan_layout_model = QVBoxLayout()
        wan_header = QHBoxLayout()
        wan_header.addWidget(QLabel("Wan S2V:"))
        self.wan_status_label = QLabel("Not Loaded")
        self.wan_status_label.setStyleSheet("color: #ff6b6b;")
        wan_header.addWidget(self.wan_status_label)
        wan_header.addStretch()
        wan_layout_model.addLayout(wan_header)
        wan_btn_layout = QHBoxLayout()
        self.wan_load_btn = QPushButton("Load")
        self.wan_load_btn.setMaximumWidth(60)
        self.wan_load_btn.clicked.connect(self.load_wan_model)
        wan_btn_layout.addWidget(self.wan_load_btn)
        self.wan_unload_btn = QPushButton("Unload")
        self.wan_unload_btn.setMaximumWidth(60)
        self.wan_unload_btn.clicked.connect(self.unload_wan_model)
        self.wan_unload_btn.setEnabled(False)
        wan_btn_layout.addWidget(self.wan_unload_btn)
        wan_layout_model.addLayout(wan_btn_layout)
        models_layout.addLayout(wan_layout_model)

        layout.addWidget(models_group)

        # Voice selection section
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Reference Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.currentTextChanged.connect(self.on_voice_changed)
        voice_layout.addWidget(self.voice_combo, stretch=1)
        self.refresh_voices_btn = QPushButton("Refresh")
        self.refresh_voices_btn.clicked.connect(self.refresh_voices)
        voice_layout.addWidget(self.refresh_voices_btn)
        self.no_reference_checkbox = QCheckBox("No reference voice")
        self.no_reference_checkbox.stateChanged.connect(self.on_no_reference_toggled)
        voice_layout.addWidget(self.no_reference_checkbox)
        layout.addLayout(voice_layout)

        # Transcribe button
        transcribe_layout = QHBoxLayout()
        self.transcribe_btn = QPushButton("Transcribe Reference Audio")
        self.transcribe_btn.clicked.connect(self.transcribe_audio)
        self.transcribe_btn.setEnabled(False)
        transcribe_layout.addWidget(self.transcribe_btn)
        transcribe_layout.addStretch()
        layout.addLayout(transcribe_layout)

        # Reference text section
        layout.addWidget(QLabel("Reference Text:"))
        self.ref_text_edit = QTextEdit()
        self.ref_text_edit.setMaximumHeight(100)
        self.ref_text_edit.setPlaceholderText("Enter or transcribe the text spoken in the reference audio...")
        layout.addWidget(self.ref_text_edit)

        # Language selection
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Target Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Italian", "Spanish", "French", "German", "Chinese", "Japanese", "Korean"])
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        layout.addLayout(lang_layout)

        # Text to generate section
        layout.addWidget(QLabel("Text to Generate:"))
        self.generate_text_edit = QTextEdit()
        self.generate_text_edit.setMaximumHeight(100)
        self.generate_text_edit.setPlaceholderText("Enter the text you want to generate with the cloned voice...")
        layout.addWidget(self.generate_text_edit)

        # Generate button
        self.generate_btn = QPushButton("Generate Audio")
        self.generate_btn.clicked.connect(self.generate_audio)
        layout.addWidget(self.generate_btn)

        # Avatar gallery header
        avatar_header = QHBoxLayout()
        avatar_header.addWidget(QLabel("Avatar Image:"))
        avatar_header.addStretch()
        self.refresh_avatars_btn = QPushButton("Refresh")
        self.refresh_avatars_btn.clicked.connect(self.refresh_avatars)
        avatar_header.addWidget(self.refresh_avatars_btn)
        layout.addLayout(avatar_header)

        # Avatar horizontal scroll gallery
        self.avatar_scroll = QScrollArea()
        self.avatar_scroll.setWidgetResizable(True)
        self.avatar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.avatar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.avatar_scroll.setFixedHeight(120)

        self.avatar_gallery_widget = QWidget()
        self.avatar_gallery_layout = QHBoxLayout(self.avatar_gallery_widget)
        self.avatar_gallery_layout.setContentsMargins(5, 5, 5, 5)
        self.avatar_gallery_layout.setSpacing(10)
        self.avatar_scroll.setWidget(self.avatar_gallery_widget)
        layout.addWidget(self.avatar_scroll)

        self.selected_avatar_path = None
        self.avatar_labels = []

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(QLabel("Audio File:"))
        self.video_audio_combo = QComboBox()
        audio_layout.addWidget(self.video_audio_combo, stretch=1)
        self.refresh_video_audio_btn = QPushButton("Refresh")
        self.refresh_video_audio_btn.clicked.connect(self.refresh_video_audio_list)
        audio_layout.addWidget(self.refresh_video_audio_btn)
        layout.addLayout(audio_layout)

        # Audio padding options
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("Padding:"))
        padding_layout.addWidget(QLabel("Before:"))
        self.padding_before_spin = QDoubleSpinBox()
        self.padding_before_spin.setRange(0, 10)
        self.padding_before_spin.setValue(0.5)
        self.padding_before_spin.setSuffix("s")
        self.padding_before_spin.setSingleStep(0.1)
        padding_layout.addWidget(self.padding_before_spin)
        padding_layout.addWidget(QLabel("After:"))
        self.padding_after_spin = QDoubleSpinBox()
        self.padding_after_spin.setRange(0, 10)
        self.padding_after_spin.setValue(0.5)
        self.padding_after_spin.setSuffix("s")
        self.padding_after_spin.setSingleStep(0.1)
        padding_layout.addWidget(self.padding_after_spin)
        padding_layout.addStretch()
        layout.addLayout(padding_layout)

        # Wan2.2 S2V generation
        wan_layout = QHBoxLayout()
        wan_layout.addWidget(QLabel("Prompt (optional):"))
        self.wan_prompt_edit = QLineEdit()
        self.wan_prompt_edit.setPlaceholderText("Describe the video style/scene (leave empty for default)")
        wan_layout.addWidget(self.wan_prompt_edit)
        layout.addLayout(wan_layout)

        self.generate_wan_video_btn = QPushButton("Generate Avatar Video")
        self.generate_wan_video_btn.clicked.connect(self.generate_wan_video)
        layout.addWidget(self.generate_wan_video_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Initial refresh
        self.refresh_voices()
        self.refresh_avatars()
        self.refresh_video_audio_list()
        self.refresh_history_list()
        self.refresh_audio_library()
        self.refresh_video_library()

    def refresh_voices(self):
        self.voice_combo.clear()
        voices_dir = Path("voices")
        if voices_dir.exists():
            voice_files = sorted(voices_dir.glob("*.wav"))
            for voice_file in voice_files:
                self.voice_combo.addItem(voice_file.name, str(voice_file))

        if self.voice_combo.count() == 0:
            self.status_label.setText("No voice files found in 'voices' directory")

    def on_voice_changed(self, voice_name):
        if voice_name:
            self.current_voice_path = self.voice_combo.currentData()
            self.transcribe_btn.setEnabled(not self.no_reference_checkbox.isChecked())

    def on_no_reference_toggled(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.voice_combo.setEnabled(not is_checked)
        self.refresh_voices_btn.setEnabled(not is_checked)
        self.transcribe_btn.setEnabled(not is_checked and self.current_voice_path is not None)
        self.ref_text_edit.setEnabled(not is_checked)
        if is_checked:
            self.current_voice_path = None

    def transcribe_audio(self):
        if not self.current_voice_path:
            QMessageBox.warning(self, "Warning", "Please select a voice file first")
            return

        self.status_label.setText("Transcribing audio...")
        self.progress_bar.setVisible(True)
        self.transcribe_btn.setEnabled(False)

        self.transcribe_thread = TranscribeThread(self.current_voice_path)
        self.transcribe_thread.finished.connect(self.on_transcribe_finished)
        self.transcribe_thread.error.connect(self.on_transcribe_error)
        self.transcribe_thread.start()

    def on_transcribe_finished(self, text):
        self.ref_text_edit.setPlainText(text)
        self.status_label.setText("Transcription complete")
        self.progress_bar.setVisible(False)
        self.transcribe_btn.setEnabled(True)
        # Update whisper status since it was loaded during transcription
        if is_whisper_loaded():
            self.whisper_status_label.setText("Loaded")
            self.whisper_status_label.setStyleSheet("color: #51cf66;")
            self.whisper_load_btn.setEnabled(False)
            self.whisper_unload_btn.setEnabled(True)

    def on_transcribe_error(self, error):
        QMessageBox.critical(self, "Transcription Error", f"Failed to transcribe audio:\n{error}\n\nMake sure whisper is installed: pip install openai-whisper")
        self.status_label.setText("Transcription failed")
        self.progress_bar.setVisible(False)
        self.transcribe_btn.setEnabled(True)

    def load_model(self):
        """Load Qwen TTS model (called automatically on generate)"""
        return self.load_qwen_model()

    def load_qwen_model(self):
        """Load Qwen TTS model"""
        self.status_label.setText("Loading Qwen TTS model...")
        self.progress_bar.setVisible(True)
        self.qwen_load_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            self.qwen_status_label.setText("Loaded")
            self.qwen_status_label.setStyleSheet("color: #51cf66;")
            self.qwen_load_btn.setEnabled(False)
            self.qwen_unload_btn.setEnabled(True)
            self.status_label.setText("Qwen TTS model loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Qwen model:\n{str(e)}")
            self.status_label.setText("Failed to load Qwen model")
            self.qwen_load_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            return False
        finally:
            self.progress_bar.setVisible(False)

        return True

    def unload_qwen_model(self):
        """Unload Qwen TTS model"""
        if self.model:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            self.qwen_status_label.setText("Not Loaded")
            self.qwen_status_label.setStyleSheet("color: #ff6b6b;")
            self.qwen_load_btn.setEnabled(True)
            self.qwen_unload_btn.setEnabled(False)
            self.status_label.setText("Qwen TTS model unloaded")

    def load_whisper_model(self):
        """Load Whisper model"""
        self.status_label.setText("Loading Whisper model...")
        self.progress_bar.setVisible(True)
        self.whisper_load_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            load_whisper_model("medium")
            self.whisper_status_label.setText("Loaded")
            self.whisper_status_label.setStyleSheet("color: #51cf66;")
            self.whisper_load_btn.setEnabled(False)
            self.whisper_unload_btn.setEnabled(True)
            self.status_label.setText("Whisper model loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Whisper model:\n{str(e)}")
            self.status_label.setText("Failed to load Whisper model")
            self.whisper_load_btn.setEnabled(True)
        finally:
            self.progress_bar.setVisible(False)

    def unload_whisper_model(self):
        """Unload Whisper model"""
        unload_whisper_model()
        self.whisper_status_label.setText("Not Loaded")
        self.whisper_status_label.setStyleSheet("color: #ff6b6b;")
        self.whisper_load_btn.setEnabled(True)
        self.whisper_unload_btn.setEnabled(False)
        self.status_label.setText("Whisper model unloaded")

    def load_wan_model(self):
        """Load Wan S2V model"""
        self.status_label.setText("Loading Wan S2V models...")
        self.progress_bar.setVisible(True)
        self.wan_load_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            load_wan_models(use_lightning_lora=True)
            self.wan_status_label.setText("Loaded")
            self.wan_status_label.setStyleSheet("color: #51cf66;")
            self.wan_load_btn.setEnabled(False)
            self.wan_unload_btn.setEnabled(True)
            self.status_label.setText("Wan S2V models loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Wan models:\n{str(e)}")
            self.status_label.setText("Failed to load Wan models")
            self.wan_load_btn.setEnabled(True)
        finally:
            self.progress_bar.setVisible(False)

    def unload_wan_model(self):
        """Unload Wan S2V model"""
        unload_wan_models()
        self.wan_status_label.setText("Not Loaded")
        self.wan_status_label.setStyleSheet("color: #ff6b6b;")
        self.wan_load_btn.setEnabled(True)
        self.wan_unload_btn.setEnabled(False)
        self.status_label.setText("Wan S2V models unloaded")

    def generate_audio(self):
        # Auto-load model if not already loaded
        if not self.model:
            if not self.load_model():
                return  # Load failed, abort generation

        use_reference = not self.no_reference_checkbox.isChecked()

        if use_reference:
            if not self.current_voice_path:
                QMessageBox.warning(self, "Warning", "Please select a reference voice")
                return

            ref_text = self.ref_text_edit.toPlainText().strip()
            if not ref_text:
                QMessageBox.warning(self, "Warning", "Please enter reference text")
                return
        else:
            ref_text = None

        generate_text = self.generate_text_edit.toPlainText().strip()
        if not generate_text:
            QMessageBox.warning(self, "Warning", "Please enter text to generate")
            return

        instruct = None

        self.status_label.setText("Generating audio...")
        self.progress_bar.setVisible(True)
        self.generate_btn.setEnabled(False)

        self.generate_audio_thread = GenerateAudioThread(
            self.model,
            generate_text,
            self.language_combo.currentText(),
            self.current_voice_path if use_reference else None,
            ref_text,
            instruct
        )
        self.generate_audio_thread.finished.connect(self.on_generate_audio_finished)
        self.generate_audio_thread.error.connect(self.on_generate_audio_error)
        self.generate_audio_thread.start()

    def on_generate_audio_finished(self, output_path):
        self.status_label.setText(f"Audio generated: {output_path}")
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)

        # Save to history
        self.add_to_history({
            'text': self.generate_text_edit.toPlainText().strip(),
            'language': self.language_combo.currentText(),
            'voice': Path(self.current_voice_path).name if self.current_voice_path else None,
            'ref_text': self.ref_text_edit.toPlainText().strip() if not self.no_reference_checkbox.isChecked() else None,
            'timestamp': datetime.now().isoformat(),
            'output': output_path
        })

        # Refresh audio library and video audio selector, then auto-play
        self.refresh_audio_library()
        self.refresh_video_audio_list()
        self.load_audio(output_path)
        self.player.play()

    def on_generate_audio_error(self, error):
        QMessageBox.critical(self, "Error", f"Failed to generate audio:\n{error}")
        self.status_label.setText("Generation failed")
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)


    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []

    def save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_to_history(self, entry):
        self.history.insert(0, entry)
        if len(self.history) > 100:
            self.history = self.history[:100]
        self.save_history()
        self.refresh_history_list()

    def refresh_history_list(self):
        self.history_list.clear()
        for i, entry in enumerate(self.history):
            text_preview = entry['text'][:50] + ('...' if len(entry['text']) > 50 else '')
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M")
            item_text = f"{timestamp} | {text_preview}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.history_list.addItem(item)

    def on_history_clicked(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx < len(self.history):
            entry = self.history[idx]

            # Load text
            self.generate_text_edit.setPlainText(entry['text'])

            # Load language
            lang_index = self.language_combo.findText(entry['language'])
            if lang_index >= 0:
                self.language_combo.setCurrentIndex(lang_index)

            # Load voice if present
            if entry.get('voice'):
                voice_index = self.voice_combo.findText(entry['voice'])
                if voice_index >= 0:
                    self.voice_combo.setCurrentIndex(voice_index)
                    self.no_reference_checkbox.setChecked(False)
                else:
                    self.no_reference_checkbox.setChecked(True)
            else:
                self.no_reference_checkbox.setChecked(True)

            # Load ref text
            if entry.get('ref_text'):
                self.ref_text_edit.setPlainText(entry['ref_text'])
            else:
                self.ref_text_edit.clear()

            self.status_label.setText(f"Loaded from history: {entry.get('output', 'N/A')}")

    def delete_history_item(self):
        current_item = self.history_list.currentItem()
        if current_item:
            idx = current_item.data(Qt.ItemDataRole.UserRole)
            if idx < len(self.history):
                del self.history[idx]
                self.save_history()
                self.refresh_history_list()

    def clear_history(self):
        reply = QMessageBox.question(
            self, 'Clear History',
            'Are you sure you want to clear all history?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.history = []
            self.save_history()
            self.refresh_history_list()

    def refresh_audio_library(self):
        self.audio_library_list.clear()
        output_dir = Path("output_voices")
        if output_dir.exists():
            audio_files = sorted(output_dir.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
            for audio_file in audio_files:
                # Get file info
                stat = audio_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                item_text = f"{audio_file.name} ({size_mb:.1f}MB) - {mod_time}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, str(audio_file))
                self.audio_library_list.addItem(item)

    def on_audio_library_clicked(self, item):
        audio_path = item.data(Qt.ItemDataRole.UserRole)
        self.load_audio(audio_path)

    def on_audio_library_double_clicked(self, item):
        audio_path = item.data(Qt.ItemDataRole.UserRole)
        self.load_audio(audio_path)
        self.player.play()

    def load_audio(self, audio_path):
        self.player.setSource(QUrl.fromLocalFile(audio_path))
        self.now_playing_label.setText(f"Loaded: {Path(audio_path).name}")
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop_playback(self):
        self.player.stop()

    def on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def on_position_changed(self, position):
        self.position_slider.setValue(position)
        self.update_time_label()

    def on_duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.update_time_label()

    def set_position(self, position):
        self.player.setPosition(position)

    def update_time_label(self):
        position = self.player.position()
        duration = self.player.duration()

        pos_min = position // 60000
        pos_sec = (position % 60000) // 1000
        dur_min = duration // 60000
        dur_sec = (duration % 60000) // 1000

        self.time_label.setText(f"{pos_min}:{pos_sec:02d} / {dur_min}:{dur_sec:02d}")

    def change_volume(self, value):
        self.audio_output.setVolume(value / 100.0)

    def delete_audio_file(self):
        current_item = self.audio_library_list.currentItem()
        if not current_item:
            return

        audio_path = current_item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, 'Delete Audio File',
            f'Are you sure you want to delete:\n{Path(audio_path).name}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Stop playback if this file is playing
                if self.player.source() == QUrl.fromLocalFile(audio_path):
                    self.player.stop()
                    self.now_playing_label.setText("No audio loaded")
                    self.play_btn.setEnabled(False)
                    self.stop_btn.setEnabled(False)

                os.remove(audio_path)
                self.refresh_audio_library()
                self.status_label.setText(f"Deleted: {Path(audio_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete file:\n{e}")

    def refresh_avatars(self):
        # Clear existing avatars
        for label in self.avatar_labels:
            label.deleteLater()
        self.avatar_labels.clear()

        avatars_dir = Path("avatars")
        if not avatars_dir.exists():
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        avatar_files = sorted(avatars_dir.glob("*.*"))

        first_avatar = None
        for avatar_file in avatar_files:
            if avatar_file.suffix.lower() not in image_extensions:
                continue

            avatar_path = str(avatar_file)
            if first_avatar is None:
                first_avatar = avatar_path

            # Create clickable thumbnail
            label = QLabel()
            label.setFixedSize(100, 100)
            label.setScaledContents(True)
            label.setStyleSheet("border: 3px solid transparent; border-radius: 5px;")
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.setToolTip(avatar_file.name)

            # Load and set pixmap
            pixmap = QPixmap(avatar_path)
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            # Store path as property
            label.setProperty("avatar_path", avatar_path)

            # Connect click event
            label.mousePressEvent = lambda e, p=avatar_path, l=label: self._select_avatar(p, l)

            self.avatar_gallery_layout.addWidget(label)
            self.avatar_labels.append(label)

        # Add stretch at the end
        self.avatar_gallery_layout.addStretch()

        # Select first avatar by default
        if first_avatar and self.avatar_labels:
            self._select_avatar(first_avatar, self.avatar_labels[0])

    def _select_avatar(self, avatar_path: str, label: QLabel):
        """Select an avatar from the gallery"""
        self.selected_avatar_path = avatar_path

        # Update visual selection
        for lbl in self.avatar_labels:
            lbl.setStyleSheet("border: 3px solid transparent; border-radius: 5px;")

        label.setStyleSheet("border: 3px solid #4dabf7; border-radius: 5px;")

    def refresh_video_audio_list(self):
        self.video_audio_combo.clear()
        output_dir = Path("output_voices")
        if output_dir.exists():
            audio_files = sorted(output_dir.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
            for audio_file in audio_files:
                self.video_audio_combo.addItem(audio_file.name, str(audio_file))

    def _pad_audio(self, audio_path: str, padding_before: float, padding_after: float) -> str:
        """Add silence padding before and after audio file"""
        import wave
        import struct
        from pathlib import Path

        with wave.open(audio_path, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)

        sample_rate = params.framerate
        sample_width = params.sampwidth
        channels = params.nchannels

        # Create silence
        silence_before = b'\x00' * int(padding_before * sample_rate * sample_width * channels)
        silence_after = b'\x00' * int(padding_after * sample_rate * sample_width * channels)

        # Create padded audio
        padded_frames = silence_before + frames + silence_after

        # Save to temp file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        padded_path = str(temp_dir / f"padded_{Path(audio_path).name}")

        with wave.open(padded_path, 'wb') as wav:
            wav.setparams(params)
            wav.setnframes(len(padded_frames) // (sample_width * channels))
            wav.writeframes(padded_frames)

        return padded_path

    def generate_wan_video(self):
        if not self.selected_avatar_path:
            QMessageBox.warning(self, "Warning", "No avatar selected. Add images to 'avatars' directory.")
            return

        if self.video_audio_combo.count() == 0:
            QMessageBox.warning(self, "Warning", "No audio files found in 'output_voices' directory")
            return

        avatar_path = self.selected_avatar_path
        audio_path = self.video_audio_combo.currentData()
        prompt = self.wan_prompt_edit.text().strip() or None

        # Apply audio padding if needed
        padding_before = self.padding_before_spin.value()
        padding_after = self.padding_after_spin.value()

        if padding_before > 0 or padding_after > 0:
            self.status_label.setText("Adding audio padding...")
            QApplication.processEvents()
            audio_path = self._pad_audio(audio_path, padding_before, padding_after)

        self.status_label.setText("Generating Wan2.2 S2V video (this may take several minutes)...")
        self.progress_bar.setVisible(True)
        self.generate_wan_video_btn.setEnabled(False)

        self.wan_video_thread = GenerateVideoThread(
            avatar_path=avatar_path,
            audio_path=audio_path,
            prompt=prompt or "A person speaking naturally with realistic facial movements.",
            use_wan_s2v=True,
            use_lightning_lora=True,
        )
        self.wan_video_thread.finished.connect(self.on_wan_video_generate_finished)
        self.wan_video_thread.error.connect(self.on_wan_video_generate_error)
        self.wan_video_thread.progress.connect(self.on_wan_video_progress)
        self.wan_video_thread.start()

    def on_wan_video_progress(self, current, total, message):
        self.status_label.setText(f"[{current}/{total}] {message}")
        if total > 0:
            self.progress_bar.setValue(int(current * 100 / total))

    def on_wan_video_generate_finished(self, output_path):
        self.status_label.setText(f"Wan2.2 video generated: {output_path}")
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.generate_wan_video_btn.setEnabled(True)
        # Update wan status since it was loaded during generation
        if get_wan_model_cache().is_loaded():
            self.wan_status_label.setText("Loaded")
            self.wan_status_label.setStyleSheet("color: #51cf66;")
            self.wan_load_btn.setEnabled(False)
            self.wan_unload_btn.setEnabled(True)

        # Refresh video library and auto-play
        self.refresh_video_library()
        self.load_video(output_path)
        self.video_player.play()

    def on_wan_video_generate_error(self, error):
        QMessageBox.critical(self, "Error", f"Failed to generate Wan2.2 video:\n{error}")
        self.status_label.setText("Wan2.2 video generation failed")
        self.progress_bar.setVisible(False)
        self.generate_wan_video_btn.setEnabled(True)

    def refresh_video_library(self):
        self.video_library_list.clear()
        output_dir = Path("output_avatars")
        if output_dir.exists():
            video_files = sorted(output_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
            for video_file in video_files:
                # Get file info
                stat = video_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

                item_text = f"{video_file.name} ({size_mb:.1f}MB) - {mod_time}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, str(video_file))
                self.video_library_list.addItem(item)

    def on_video_library_clicked(self, item):
        video_path = item.data(Qt.ItemDataRole.UserRole)
        self.load_video(video_path)

    def on_video_library_double_clicked(self, item):
        video_path = item.data(Qt.ItemDataRole.UserRole)
        self.load_video(video_path)
        self.video_player.play()

    def load_video(self, video_path):
        self.video_player.setSource(QUrl.fromLocalFile(video_path))
        self.video_now_playing_label.setText(f"Loaded: {Path(video_path).name}")
        self.video_play_btn.setEnabled(True)
        self.video_stop_btn.setEnabled(True)

    def toggle_video_playback(self):
        if self.video_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.video_player.pause()
        else:
            self.video_player.play()

    def stop_video_playback(self):
        self.video_player.stop()

    def on_video_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.video_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.video_play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def on_video_position_changed(self, position):
        self.video_position_slider.setValue(position)
        self.update_video_time_label()

    def on_video_duration_changed(self, duration):
        self.video_position_slider.setRange(0, duration)
        self.update_video_time_label()

    def set_video_position(self, position):
        self.video_player.setPosition(position)

    def update_video_time_label(self):
        position = self.video_player.position()
        duration = self.video_player.duration()

        pos_min = position // 60000
        pos_sec = (position % 60000) // 1000
        dur_min = duration // 60000
        dur_sec = (duration % 60000) // 1000

        self.video_time_label.setText(f"{pos_min}:{pos_sec:02d} / {dur_min}:{dur_sec:02d}")

    def change_video_volume(self, value):
        self.video_audio_output.setVolume(value / 100.0)

    def delete_video_file(self):
        current_item = self.video_library_list.currentItem()
        if not current_item:
            return

        video_path = current_item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, 'Delete Video File',
            f'Are you sure you want to delete:\n{Path(video_path).name}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Stop playback if this file is playing
                if self.video_player.source() == QUrl.fromLocalFile(video_path):
                    self.video_player.stop()
                    self.video_now_playing_label.setText("No video loaded")
                    self.video_play_btn.setEnabled(False)
                    self.video_stop_btn.setEnabled(False)

                os.remove(video_path)
                self.refresh_video_library()
                self.status_label.setText(f"Deleted: {Path(video_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete file:\n{e}")


def main():
    app = QApplication(sys.argv)
    window = TTSMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
