#!/usr/bin/env python3
"""
Wan2.2 S2V Video Generation - CLI wrapper

This script uses the worker module which keeps models cached in memory
for faster subsequent generations.
"""

import sys
from pathlib import Path

# Import from worker module
from workers.generate_video_worker import generate_video, load_models, unload_models


if __name__ == "__main__":
    # Default paths
    image_path = Path("avatars/sydney.png")
    audio_path = Path("output_voices/sydney_2026-02-05T19.31.52_padded.wav")

    # Find first available image
    if not image_path.exists():
        avatars_dir = Path("avatars")
        if avatars_dir.exists():
            imgs = list(avatars_dir.glob("*.png")) + list(avatars_dir.glob("*.jpg"))
            if imgs:
                image_path = imgs[0]

    # Find first available audio
    if not audio_path.exists():
        voices_dir = Path("output_voices")
        if voices_dir.exists():
            auds = list(voices_dir.glob("*_padded.wav")) + list(voices_dir.glob("*.wav"))
            if auds:
                audio_path = auds[0]

    if not image_path.exists() or not audio_path.exists():
        print("Error: Missing input files")
        print(f"  Image: {image_path} (exists: {image_path.exists()})")
        print(f"  Audio: {audio_path} (exists: {audio_path.exists()})")
        sys.exit(1)

    try:
        output = generate_video(
            avatar_path=str(image_path),
            audio_path=str(audio_path),
            output_path="output_avatars/" + image_path.stem + "_s2v.mp4",
            use_lightning_lora=True,
        )
        print(f"\nSuccess! {output}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
