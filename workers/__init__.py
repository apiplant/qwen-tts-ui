# Workers package - PyQt6 GUI thread workers
# These require PyQt6 to be installed

try:
    from workers.transcribe_worker import TranscribeThread
    from workers.generate_audio_worker import GenerateAudioThread
    from workers.generate_video_worker import GenerateVideoThread

    __all__ = ['TranscribeThread', 'GenerateAudioThread', 'GenerateVideoThread']
except ImportError:
    # PyQt6 not available - workers won't be available but module can still be imported
    __all__ = []

# Always export the non-PyQt6 functions
try:
    from workers.generate_video_worker import (
        generate_video,
        load_models as load_wan_models,
        unload_models as unload_wan_models,
        get_model_cache as get_wan_model_cache,
    )
    __all__.extend(['generate_video', 'load_wan_models', 'unload_wan_models', 'get_wan_model_cache'])
except ImportError:
    pass

try:
    from workers.transcribe_worker import (
        load_whisper_model,
        unload_whisper_model,
        is_whisper_loaded,
        transcribe,
    )
    __all__.extend(['load_whisper_model', 'unload_whisper_model', 'is_whisper_loaded', 'transcribe'])
except ImportError:
    pass
