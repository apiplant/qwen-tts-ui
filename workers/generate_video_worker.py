"""
Wan2.2 S2V Video Generation Worker

Generates video from image + audio using Wan2.2 S2V model.
Keeps models loaded in memory for faster subsequent generations.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import random

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set models path before importing comfy
os.environ['COMFY_MODELS_PATH'] = str(PROJECT_ROOT / 'models')

import torch
import numpy as np
from PIL import Image
import av

# PyQt6 is optional - only needed for GUI thread
try:
    from PyQt6.QtCore import QThread, pyqtSignal
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    QThread = object
    pyqtSignal = lambda *args: None

# Constants
FPS = 16
CHUNK_FRAMES = 77  # Wan2.2 minimum/chunk size


class WanS2VModelCache:
    """
    Singleton cache for Wan2.2 S2V models.
    Keeps models loaded in memory for faster subsequent generations.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if WanS2VModelCache._initialized:
            return
        WanS2VModelCache._initialized = True

        self.clip = None
        self.vae = None
        self.model = None
        self.audio_encoder = None
        self.use_lightning_lora = None

        # Node instances
        self._nodes_initialized = False

    def _init_nodes(self):
        """Initialize node instances (lazy loading)"""
        if self._nodes_initialized:
            return

        from comfy.nodes.base_nodes import (
            CLIPLoader, VAELoader, UNETLoader, CLIPTextEncode,
            KSampler, VAEDecode, LoraLoaderModelOnly,
        )
        from comfy_extras.nodes.nodes_audio import LoadAudio
        from comfy_extras.nodes.nodes_audio_encoder import AudioEncoderLoader, AudioEncoderEncode
        from comfy_extras.nodes.nodes_wan import WanSoundImageToVideo, WanSoundImageToVideoExtend
        from comfy_extras.nodes.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes.nodes_latent import LatentConcat

        self.clip_loader = CLIPLoader()
        self.vae_loader = VAELoader()
        self.unet_loader = UNETLoader()
        self.clip_encode = CLIPTextEncode()
        self.ksampler = KSampler()
        self.vae_decode = VAEDecode()
        self.lora_loader = LoraLoaderModelOnly()
        self.load_audio = LoadAudio()
        self.audio_encoder_loader = AudioEncoderLoader()
        self.audio_encoder_encode = AudioEncoderEncode()
        self.wan_s2v = WanSoundImageToVideo()
        self.wan_s2v_extend = WanSoundImageToVideoExtend()
        self.model_sampling = ModelSamplingSD3()
        self.latent_concat = LatentConcat()

        self._nodes_initialized = True

    def load_models(
        self,
        use_lightning_lora: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Load all models into memory.

        Args:
            use_lightning_lora: Whether to use 4-step lightning lora
            progress_callback: Optional callback for progress updates
        """
        def log(msg: str):
            print(msg)
            if progress_callback:
                progress_callback(msg)

        # Check if already loaded with same settings
        if (self.clip is not None and
            self.vae is not None and
            self.model is not None and
            self.audio_encoder is not None and
            self.use_lightning_lora == use_lightning_lora):
            log("Models already loaded, reusing cached models")
            return

        self._init_nodes()

        with torch.inference_mode():
            log("Loading CLIP (umt5_xxl)...")
            self.clip = self.clip_loader.load_clip(
                clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                type="wan"
            )[0]

            log("Loading VAE (wan_2.1)...")
            self.vae = self.vae_loader.load_vae(
                vae_name="wan_2.1_vae.safetensors"
            )[0]

            log("Loading UNET (wan2.2_s2v_14B)...")
            model = self.unet_loader.load_unet(
                unet_name="wan2.2_s2v_14B_fp8_scaled.safetensors",
                weight_dtype="default"
            )[0]

            log("Loading Audio Encoder (wav2vec2)...")
            self.audio_encoder = self.audio_encoder_loader.execute(
                audio_encoder_name="wav2vec2_large_english_fp16.safetensors"
            )[0]

            if use_lightning_lora:
                log("Loading Lightning LoRA (4-step)...")
                model = self.lora_loader.load_lora_model_only(
                    model=model,
                    lora_name="wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
                    strength_model=1.0
                )[0]

            log("Applying model sampling (shift=8)...")
            self.model = self.model_sampling.patch(model=model, shift=8.0)[0]
            self.use_lightning_lora = use_lightning_lora

            log("All models loaded!")

    def unload_models(self):
        """Unload all models from memory"""
        self.clip = None
        self.vae = None
        self.model = None
        self.audio_encoder = None
        self.use_lightning_lora = None
        torch.cuda.empty_cache()
        print("Models unloaded")

    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return all([
            self.clip is not None,
            self.vae is not None,
            self.model is not None,
            self.audio_encoder is not None
        ])


# Global model cache instance
_model_cache = WanS2VModelCache()


def get_model_cache() -> WanS2VModelCache:
    """Get the global model cache instance"""
    return _model_cache


def load_models(use_lightning_lora: bool = True, progress_callback: Optional[Callable[[str], None]] = None):
    """
    Pre-load models into memory for faster generation.

    Args:
        use_lightning_lora: Whether to use 4-step lightning lora (faster)
        progress_callback: Optional callback for progress updates
    """
    _model_cache.load_models(use_lightning_lora, progress_callback)


def unload_models():
    """Unload models from memory"""
    _model_cache.unload_models()


def _load_image_tensor(image_path: str) -> torch.Tensor:
    """Load image as ComfyUI tensor [B, H, W, C]"""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)[None,]


def generate_video(
    avatar_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    prompt: str = "A person speaking naturally with realistic facial movements.",
    negative_prompt: str = "static, blurry, low quality, frozen frame, distorted, deformed",
    width: int = 640,
    height: int = 640,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = -1,
    use_lightning_lora: bool = True,
    use_wan_s2v: bool = True,
    trim_start: float = 0.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Generate video from image and audio using Wan2.2 S2V.

    Args:
        avatar_path: Path to reference image
        audio_path: Path to audio file
        output_path: Optional output path (auto-generated if not provided)
        prompt: Positive prompt describing the video
        negative_prompt: Negative prompt
        width: Video width (default 640)
        height: Video height (default 640)
        steps: Number of diffusion steps (4 with lightning, 20 without)
        cfg: CFG scale (1.0 with lightning, 6.0 without)
        seed: Random seed (-1 for random)
        use_lightning_lora: Use 4-step lightning lora for faster generation
        use_wan_s2v: Use Wan S2V model (False for static video fallback)
        trim_start: Seconds to trim from start of output video
        progress_callback: Optional callback(current_step, total_steps, message)

    Returns:
        Path to generated video file
    """
    def log(step: int, total: int, msg: str):
        print(f"[{step}/{total}] {msg}")
        if progress_callback:
            progress_callback(step, total, msg)

    if not use_wan_s2v:
        return _generate_static_video(avatar_path, audio_path, output_path)

    # Validate inputs
    avatar_path = str(Path(avatar_path).resolve())
    audio_path = str(Path(audio_path).resolve())

    if not Path(avatar_path).exists():
        raise FileNotFoundError(f"Image not found: {avatar_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Generate output path
    if output_path is None:
        avatar_name = Path(avatar_path).stem
        timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
        output_path = f"output_avatars/{avatar_name}_{timestamp}_s2v.mp4"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Adjust params for non-lightning mode
    if not use_lightning_lora:
        steps = 20
        cfg = 6.0

    # Set seed
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # Ensure models are loaded
    cache = get_model_cache()
    if not cache.is_loaded() or cache.use_lightning_lora != use_lightning_lora:
        log(0, 10, "Loading models...")
        cache.load_models(use_lightning_lora, lambda msg: log(0, 10, msg))

    log(1, 10, "Loading inputs...")

    with torch.inference_mode():
        # Load image
        image = _load_image_tensor(avatar_path)

        # Load audio
        audio = cache.load_audio.execute(audio=audio_path)[0]

        # Calculate required frames from audio duration
        audio_data = audio.get('waveform', None)
        sample_rate = audio.get('sample_rate', 44100)
        if audio_data is not None:
            audio_samples = audio_data.shape[-1]
            audio_duration = audio_samples / sample_rate
            required_frames = int(audio_duration * FPS) + 1
        else:
            audio_duration = CHUNK_FRAMES / FPS
            required_frames = CHUNK_FRAMES

        num_chunks = max(1, (required_frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES)
        total_frames = num_chunks * CHUNK_FRAMES

        log(2, 10, f"Audio: {audio_duration:.2f}s, generating {num_chunks} chunk(s)...")

        # Encode text
        log(3, 10, "Encoding prompts...")
        positive = cache.clip_encode.encode(text=prompt, clip=cache.clip)[0]
        negative = cache.clip_encode.encode(text=negative_prompt, clip=cache.clip)[0]

        # Encode audio
        log(4, 10, "Encoding audio...")
        audio_embeds = cache.audio_encoder_encode.execute(
            audio_encoder=cache.audio_encoder,
            audio=audio
        )[0]

        # Generate first chunk (with +1 frame hack to avoid overbaked first frame)
        # We generate one extra frame and remove it later
        log(5, 10, f"Generating chunk 1/{num_chunks}...")
        s2v_result = cache.wan_s2v.execute(
            positive=positive,
            negative=negative,
            vae=cache.vae,
            width=width,
            height=height,
            length=CHUNK_FRAMES + 1,  # +1 for first frame hack
            batch_size=1,
            audio_encoder_output=audio_embeds,
            ref_image=image,
        )
        pos_cond = s2v_result[0]
        neg_cond = s2v_result[1]
        latent = s2v_result[2]

        samples = cache.ksampler.sample(
            model=cache.model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name="uni_pc",
            scheduler="simple",
            positive=pos_cond,
            negative=neg_cond,
            latent_image=latent,
            denoise=1.0
        )[0]

        # Extend for additional chunks
        for chunk_idx in range(1, num_chunks):
            log(5 + chunk_idx, 10, f"Generating chunk {chunk_idx + 1}/{num_chunks}...")

            extend_result = cache.wan_s2v_extend.execute(
                positive=positive,
                negative=negative,
                vae=cache.vae,
                length=CHUNK_FRAMES,
                video_latent=samples,
                audio_encoder_output=audio_embeds,
                ref_image=image,
            )
            ext_pos = extend_result[0]
            ext_neg = extend_result[1]
            ext_latent = extend_result[2]

            ext_samples = cache.ksampler.sample(
                model=cache.model,
                seed=seed + chunk_idx,
                steps=steps,
                cfg=cfg,
                sampler_name="uni_pc",
                scheduler="simple",
                positive=ext_pos,
                negative=ext_neg,
                latent_image=ext_latent,
                denoise=1.0
            )[0]

            samples = cache.latent_concat.execute(
                samples1=samples,
                samples2=ext_samples,
                dim='t'
            )[0]

        # Decode
        log(8, 10, f"Decoding {total_frames + 1} frames...")
        frames = cache.vae_decode.decode(samples=samples, vae=cache.vae)[0]

        # Remove first frame (overbaked frame hack)
        frames = frames[1:]

        # Trim to match audio duration
        if frames.shape[0] > required_frames:
            frames = frames[:required_frames]

        # Trim start of video if requested
        trim_frames = int(trim_start * FPS)
        if trim_frames > 0 and trim_frames < frames.shape[0]:
            frames = frames[trim_frames:]
            # Also trim audio to match
            trim_samples = int(trim_start * sample_rate)
            if audio_data is not None and trim_samples < audio_data.shape[-1]:
                audio_data = audio_data[..., trim_samples:]

        # Save video
        log(9, 10, "Saving video...")
        _save_video_with_audio(frames, audio_data, sample_rate, output_path)

        log(10, 10, "Done!")

        return output_path


def _save_video_with_audio(
    frames: torch.Tensor,
    audio_data: torch.Tensor,
    sample_rate: int,
    output_path: str
):
    """Save video frames with audio using PyAV"""
    frames_np = torch.clamp(frames * 255, 0, 255).to(torch.uint8).cpu().numpy()

    container = av.open(output_path, mode='w')
    video_stream = container.add_stream('h264', rate=FPS)
    video_stream.width = frames_np.shape[2]
    video_stream.height = frames_np.shape[1]
    video_stream.pix_fmt = 'yuv420p'

    # Add audio stream
    audio_stream = None
    if audio_data is not None:
        audio_stream = container.add_stream('aac', rate=sample_rate)
        audio_stream.layout = 'stereo' if audio_data.shape[1] == 2 else 'mono'

    # Write video frames
    for i in range(frames_np.shape[0]):
        frame = av.VideoFrame.from_ndarray(frames_np[i], format='rgb24')
        for packet in video_stream.encode(frame):
            container.mux(packet)

    for packet in video_stream.encode():
        container.mux(packet)

    # Write audio
    if audio_stream is not None and audio_data is not None:
        audio_np = audio_data[0].cpu().numpy()
        if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
            audio_np = (audio_np * 32767).astype(np.int16)

        frame = av.AudioFrame.from_ndarray(audio_np, format='s16p', layout=audio_stream.layout.name)
        frame.sample_rate = sample_rate
        frame.pts = 0
        for packet in audio_stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

    container.close()


def _generate_static_video(
    avatar_path: str,
    audio_path: str,
    output_path: Optional[str] = None
) -> str:
    """Fallback: Generate static video with image and audio (no AI generation)"""
    from moviepy import ImageClip, AudioFileClip

    audio_clip = AudioFileClip(audio_path)
    video = ImageClip(avatar_path, duration=audio_clip.duration)
    video = video.with_audio(audio_clip)

    if output_path is None:
        avatar_name = Path(avatar_path).stem
        timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
        output_path = f"output_avatars/{avatar_name}_{timestamp}_static.mp4"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24, logger=None)

    video.close()
    audio_clip.close()

    return output_path


class GenerateVideoThread(QThread if HAS_PYQT6 else object):
    """Thread for generating video from audio using Wan S2V model"""
    if HAS_PYQT6:
        finished = pyqtSignal(str)
        error = pyqtSignal(str)
        progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        avatar_path: str,
        audio_path: str,
        prompt: str = "A person speaking naturally with realistic facial movements.",
        steps: int = 4,
        cfg_scale: float = 1.0,
        seed: int = -1,
        use_lightning_lora: bool = True,
        use_wan_s2v: bool = True,
        width: int = 640,
        height: int = 640,
        trim_start: float = 0.0,
        **kwargs  # Accept legacy args
    ):
        super().__init__()
        self.avatar_path = avatar_path
        self.audio_path = audio_path
        self.prompt = prompt
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.use_lightning_lora = use_lightning_lora
        self.use_wan_s2v = use_wan_s2v
        self.width = width
        self.height = height
        self.trim_start = trim_start

    def run(self):
        try:
            def progress_cb(current, total, msg):
                if HAS_PYQT6:
                    self.progress.emit(current, total, msg)

            output_path = generate_video(
                avatar_path=self.avatar_path,
                audio_path=self.audio_path,
                prompt=self.prompt,
                steps=self.steps,
                cfg=self.cfg_scale,
                seed=self.seed,
                use_lightning_lora=self.use_lightning_lora,
                use_wan_s2v=self.use_wan_s2v,
                width=self.width,
                height=self.height,
                trim_start=self.trim_start,
                progress_callback=progress_cb,
            )

            if HAS_PYQT6:
                self.finished.emit(output_path)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            if HAS_PYQT6:
                self.error.emit(error_msg)
            else:
                raise


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate video from image and audio")
    parser.add_argument("--image", "-i", required=True, help="Path to reference image")
    parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--prompt", "-p", default="A person speaking naturally with realistic facial movements.")
    parser.add_argument("--steps", type=int, default=4, help="Diffusion steps (4 with lightning, 20 without)")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (1.0 with lightning, 6.0 without)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--no-lightning", action="store_true", help="Don't use lightning lora (slower but maybe better)")
    parser.add_argument("--static", action="store_true", help="Generate static video (no AI)")
    parser.add_argument("--preload", action="store_true", help="Preload models and exit")

    args = parser.parse_args()

    if args.preload:
        print("Preloading models...")
        load_models(use_lightning_lora=not args.no_lightning)
        print("Models loaded and cached. Run again to generate video.")
        sys.exit(0)

    output = generate_video(
        avatar_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        prompt=args.prompt,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        use_lightning_lora=not args.no_lightning,
        use_wan_s2v=not args.static,
    )

    print(f"\nVideo saved to: {output}")
    if Path(output).exists():
        size_mb = Path(output).stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
