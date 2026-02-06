# Qwen TTS Voice Cloner & Avatar Video Generator

A PyQt6-based application for voice cloning with Qwen TTS and talking avatar video generation with Wan2.2 S2V.

![Screenshot](screenshot.png)

## Features

### Voice Generation
- **Voice cloning**: Clone any voice from a reference audio file
- **No reference mode**: Generate speech without voice cloning
- **Whisper transcription**: Auto-transcribe reference audio
- **Multi-language support**: English, Italian, Spanish, French, German, Chinese, Japanese, Korean
- **Audio library**: Browse and play generated audio with built-in player
- **History tracking**: All generations saved with one-click reload

### Avatar Video Generation
- **Wan2.2 S2V**: Generate realistic talking avatar videos from image + audio
- **Lightning LoRA**: Fast 4-step generation (or 20-step for higher quality)
- **Auto video extension**: Automatically extends video to match audio length
- **Audio padding**: Add silence before/after audio
- **Trim start**: Remove first frames to fix overbaked artifacts
- **Video library**: Browse and play generated videos

### Model Management
- **Lazy loading**: Models load automatically when needed
- **Manual control**: Load/unload Qwen, Whisper, and Wan models independently
- **Memory efficient**: Unload unused models to free GPU memory

## Installation

Using uv (recommended):
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

## Model Setup

### Qwen TTS & Whisper
These models download automatically on first use.

### Wan2.2 S2V Video Models

Place models in the `models/` directory, or symlink from an existing ComfyUI installation:

```bash
# Option 1: Symlink from ComfyUI
ln -s /path/to/ComfyUI/models models

# Option 2: Download models manually
mkdir -p models/{text_encoders,vae,diffusion_models,audio_encoders,loras}
```

Required models (download from [Wan2.2 S2V Guide](https://docs.comfy.org/tutorials/video/wan/wan2-2-s2v)):

| Model | Directory | Filename |
|-------|-----------|----------|
| UMT5-XXL | `models/text_encoders/` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| Wan 2.1 VAE | `models/vae/` | `wan_2.1_vae.safetensors` |
| Wan2.2 S2V 14B | `models/diffusion_models/` | `wan2.2_s2v_14B_fp8_scaled.safetensors` |
| Wav2Vec2 | `models/audio_encoders/` | `wav2vec2_large_english_fp16.safetensors` |
| Lightning LoRA | `models/loras/` | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` |

## Usage

### Run the UI
```bash
uv run tts-ui
```

Or if installed:
```bash
tts-ui
```

### Voice Generation
1. Place reference voice WAV files in `voices/`
2. Select a reference voice (or check "No reference voice")
3. Transcribe or enter the reference text
4. Select target language
5. Enter text to generate
6. Click "Generate Audio"

### Video Generation
1. Place avatar images in `avatars/`
2. Select an avatar from the gallery
3. Select an audio file from the Audio Library
4. Optionally adjust padding and enter a prompt
5. Click "Generate Avatar Video"

Output files are saved to `output_voices/` (audio) and `output_avatars/` (video).

## Project Structure

```
qwen-tts/
├── voices/          # Reference voice WAV files
├── avatars/         # Avatar images for video generation
├── output_voices/   # Generated audio files
├── output_avatars/  # Generated video files
├── models/          # Wan2.2 model files (or symlink to ComfyUI)
├── ui/              # PyQt6 UI code
├── workers/         # Background workers for generation
└── pyproject.toml   # Project configuration
```

## Requirements

- Python 3.10+
- CUDA-capable GPU with sufficient VRAM
  - Qwen TTS: ~4GB
  - Whisper Medium: ~2GB
  - Wan2.2 S2V: ~20GB (fp8 quantized)

## Tips

- Use the mini player next to "Reference Text" to preview the reference voice
- Double-click files in the library to play them immediately
- Lightning LoRA is faster but slightly lower quality - disable for best results
- Trim start (default 0.2s) helps remove overbaked first frames
- Audio padding helps create natural pauses at video start/end
