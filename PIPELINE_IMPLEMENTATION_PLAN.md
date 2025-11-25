# VidLipSyncVoice Pipeline - Implementation Plan

## Overview
This document provides a detailed implementation plan for the VidLipSyncVoice pipeline that:
1. Downloads YouTube videos
2. Transcribes Vietnamese speech with timestamps using PhoWhisper
3. Synthesizes new voice using F5-TTS-Vietnamese model
4. Synchronizes lip movements using Wav2Lip
5. Composites the generated video with the original
6. Saves the final output

---

## Architecture Overview

```
┌─────────────────┐
│ YouTube Video   │
│   Download      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PhoWhisper     │
│  Transcription  │ ──► Transcription + Timestamps
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  F5-TTS Voice   │
│   Synthesis     │ ──► Generated Audio
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Wav2Lip       │
│  Lip Sync       │ ──► Synced Video
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Video       │
│  Composition    │ ──► Final Output
└─────────────────┘
```

---

## Step 1: YouTube Video Download Module

### Purpose
Download YouTube videos with separate video and audio streams for further processing.

### Implementation Status
**IMPLEMENTED** - `scripts/download_youtube_video.py`

### Implementation Details

**Library**: `yt-dlp` (modern fork of youtube-dl)

**Key Features**:
- Download video in best quality
- Extract audio separately for transcription
- Extract first frame or video segment for face detection
- Support for various video formats

**Code Structure**:
```python
# scripts/download_youtube_video.py

import yt_dlp
from pathlib import Path

def download_youtube_video(url, output_dir="input/youtube"):
    """
    Download YouTube video with separate audio and video streams.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save downloads
    
    Returns:
        dict: Paths to video, audio, and metadata
    """
    # Configuration
    - Download video (mp4, best quality)
    - Download audio (wav/mp3 for transcription)
    - Extract metadata (title, duration, uploader)
    - Save thumbnail
    
    # Return paths for pipeline
```

**Dependencies**:
```bash
pip install yt-dlp
```

**CLI Usage**:
```bash
python scripts/download_youtube_video.py <youtube_url> [output_dir]
```

**Output Structure**:
```
input/youtube/
├── video_id/
│   ├── video.mp4          # Original video
│   ├── audio.wav          # Extracted audio for transcription
│   ├── thumbnail.jpg      # Video thumbnail
│   └── metadata.json      # Video metadata
```

---

## Step 2: PhoWhisper Transcription Module

### Purpose
Transcribe Vietnamese speech from video audio with word-level timestamps for accurate synchronization.

### Implementation Status
**IMPLEMENTED** - `scripts/transcribe_phowhisper.py`

### Implementation Details

**Model**: PhoWhisper (Vietnamese-optimized Whisper model) from HuggingFace transformers

**Alternative**: OpenAI Whisper with Vietnamese language support

**Key Features**:
- ✅ Word-level timestamp extraction
- ✅ Vietnamese language optimization with PhoWhisper
- ✅ Segment-based transcription
- ✅ Support for both PhoWhisper and OpenAI Whisper models
- ✅ Flexible model path loading (local or download)
- ✅ JSON output with structured timing information
- ✅ Detailed progress reporting and segment preview

**Code Structure**:
```python
# scripts/transcribe_phowhisper.py

from transformers import pipeline  # For PhoWhisper
import whisper  # For OpenAI Whisper fallback
from pathlib import Path
import json

def transcribe_with_timestamps(audio_path, model_name="base", language="vi", 
                               verbose=True, model_path=None):
    """
    Transcribe audio with word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        model_name: Model size (tiny/base/small/medium/large) or "phowhisper"
        language: Language code (default: "vi" for Vietnamese)
        verbose: Print progress information
        model_path: Path to local PhoWhisper model directory
    
    Returns:
        dict: {
            'text': full transcription,
            'language': language code,
            'duration': audio duration in seconds,
            'audio_path': path to source audio,
            'segments': [{
                'id': segment id,
                'start': timestamp,
                'end': timestamp,
                'text': segment text,
                'words': word-level timestamps
            }]
        }
    """
```

**PhoWhisper Model Location**:
The implementation uses an existing PhoWhisper model at:
```
/home/psilab/models--vinai--PhoWhisper-medium/snapshots/55a7e3eb6c906de891f8f06a107754427dd3be79/
```

No need to move or copy the model - the script loads it directly from this location.

**Dependencies**:
```bash
# Install required packages
pip install transformers>=4.35.0
pip install torch>=2.0.0
pip install accelerate>=0.24.0

# Optional: OpenAI Whisper as fallback
pip install openai-whisper>=20231117
```

**CLI Usage**:
```bash
# Using PhoWhisper (auto-detects model at default location)
python scripts/transcribe_phowhisper.py temp/downloads/VIDEO_ID_Title/audio.wav --model phowhisper

# Using PhoWhisper with custom model path
python scripts/transcribe_phowhisper.py audio.wav --model-path /path/to/phowhisper/model

# Using OpenAI Whisper (base model)
python scripts/transcribe_phowhisper.py audio.wav --model base

# Using OpenAI Whisper (medium model for better accuracy)
python scripts/transcribe_phowhisper.py audio.wav --model medium

# Specify custom output location
python scripts/transcribe_phowhisper.py audio.wav --output transcription.json

# Show detailed segments with word timestamps
python scripts/transcribe_phowhisper.py audio.wav --show-segments 10
```

**Programmatic Usage**:
```python
from scripts.transcribe_phowhisper import transcribe_with_timestamps, save_transcription

# Using PhoWhisper with auto-detection
transcription = transcribe_with_timestamps(
    audio_path="temp/downloads/VIDEO_ID/audio.wav",
    model_name="phowhisper"
)

# Using PhoWhisper with custom path
transcription = transcribe_with_timestamps(
    audio_path="audio.wav",
    model_path="/home/psilab/models--vinai--PhoWhisper-medium/snapshots/55a7e3eb6c906de891f8f06a107754427dd3be79"
)

# Using OpenAI Whisper
transcription = transcribe_with_timestamps(
    audio_path="audio.wav",
    model_name="medium",
    language="vi"
)

# Save transcription
save_transcription(transcription, "output.json")
```

**Output Format**:
```json
{
  "text": "xin chào các bạn hôm nay chúng ta sẽ học về...",
  "language": "vi",
  "duration": 120.5,
  "audio_path": "/path/to/audio.wav",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "xin chào các bạn",
      "words": [
        {"word": "xin", "start": 0.0, "end": 0.3},
        {"word": "chào", "start": 0.4, "end": 0.8},
        {"word": "các", "start": 0.9, "end": 1.2},
        {"word": "bạn", "start": 1.3, "end": 1.6}
      ]
    },
    {
      "id": 1,
      "start": 3.5,
      "end": 7.2,
      "text": "hôm nay chúng ta sẽ học về",
      "words": [
        {"word": "hôm", "start": 3.5, "end": 3.8},
        {"word": "nay", "start": 3.9, "end": 4.2},
        {"word": "chúng", "start": 4.3, "end": 4.6},
        {"word": "ta", "start": 4.7, "end": 4.9},
        {"word": "sẽ", "start": 5.0, "end": 5.3},
        {"word": "học", "start": 5.4, "end": 5.7},
        {"word": "về", "start": 5.8, "end": 6.1}
      ]
    }
  ]
}
```

**Output Location**:
By default, transcription is saved to: `{audio_filename}_transcription.json` in the same directory as the audio file.

**Helper Functions**:
```python
# Load existing transcription
from scripts.transcribe_phowhisper import load_transcription
transcription = load_transcription("path/to/transcription.json")

# Print formatted segments
from scripts.transcribe_phowhisper import print_segments
print_segments(transcription, max_segments=5)
```

---

## Step 3: F5-TTS Voice Synthesis Module

### Purpose
Convert transcribed text to speech using target speaker's voice characteristics.

### Implementation Details

**Model**: F5-TTS-Vietnamese (Pre-trained on Vietnamese dataset)

**Model Location**: `/home/psilab/F5-TTS-Vietnamese/`
- Model checkpoint: `model/model_last.pt`
- Vocabulary: `model/vocab.txt`
- Pre-trained specifically for Vietnamese language with Vietnamese phonetics and prosody

**Key Features**:
- ✅ Pre-trained F5-TTS model optimized for Vietnamese speech
- ✅ Support multiple reference speakers from `original_voice_ref/`
- ✅ Voice cloning with minimal reference audio (5-10 seconds)
- ✅ Natural Vietnamese prosody and intonation
- ✅ Handle Vietnamese diacritics and tones correctly
- Generate speech with timing control
- Maintain speaker characteristics from reference audio

**Available Reference Speakers**:
The F5-TTS-Vietnamese project includes pre-recorded reference voices:
```python
AVAILABLE_SPEAKERS = {
    "tien_bip": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/tien_bip/",
    "kha_banh": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/kha_banh/",
    "huan_hoa_hong": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/huan_hoa_hong/",
    "quang_linh_vlog": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/quang_linh_vlog/",
    "son_tung_mtp": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/son_tung_mtp/",
    "thoi_su_nam_ha_noi": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/thoi_su_nam_ha_noi/",
    "thoi_su_nu_sai_gon": "/home/psilab/F5-TTS-Vietnamese/original_voice_ref/thoi_su_nu_sai_gon/",
    # ... and more
}
```

**Code Structure**:
```python
# scripts/synthesize_voice.py

import sys
sys.path.append('/home/psilab/F5-TTS-Vietnamese')

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
from pathlib import Path
import torch

def synthesize_speech(
    text,
    ref_audio,
    ref_text,
    output_path,
    model_path="/home/psilab/F5-TTS-Vietnamese/model/model_last.pt",
    vocab_path="/home/psilab/F5-TTS-Vietnamese/model/vocab.txt",
    speed=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Synthesize Vietnamese speech using F5-TTS-Vietnamese model.
    
    Args:
        text: Vietnamese text to synthesize
        ref_audio: Path to reference speaker audio (WAV, 5-10 seconds)
        ref_text: Transcription of reference audio (Vietnamese text)
        output_path: Where to save generated audio
        model_path: Path to F5-TTS Vietnamese checkpoint
        vocab_path: Path to Vietnamese vocabulary file
        speed: Speech speed multiplier (default: 1.0)
        device: Computation device (cuda/cpu)
    
    Returns:
        str: Path to generated audio file
    """
    # Load F5-TTS Vietnamese model
    # Process Vietnamese text (handle diacritics, tones)
    # Generate audio with reference speaker characteristics
    # Save with appropriate timing
```

**Integration with Existing F5-TTS-Vietnamese Code**:
- ✅ Reuse existing `infer.py` inference logic from F5-TTS-Vietnamese
- ✅ Leverage `src/f5_tts/infer/infer_cli.py` CLI utilities
- ✅ Use existing model loading functions from `src/f5_tts/infer/utils_infer.py`
- ✅ Support batch processing for long texts using existing chunking logic
- Access pre-trained Vietnamese model and vocabulary

**Vietnamese Text Handling**:
- Properly handle Vietnamese diacritics (á, à, ả, ã, ạ, etc.)
- Maintain tonal accuracy (6 tones in Vietnamese)
- Support Vietnamese-specific characters (ă, â, đ, ê, ô, ơ, ư)
- Handle word boundaries and prosody correctly

**CLI Usage** (using existing F5-TTS-Vietnamese CLI):
```bash
# Using existing shell scripts from script_infer folder
cd /home/psilab/F5-TTS-Vietnamese
bash script_infer/infer_tran_ha_linh.sh
```

**Existing Inference Scripts**:
The F5-TTS-Vietnamese project already has ready-to-use inference scripts in `/home/psilab/F5-TTS-Vietnamese/script_infer/`:
- `infer_tien_bip.sh` - Pre-configured for tien_bip speaker
- Other speaker-specific scripts for different reference voices

Example from `infer_tien_bip.sh`:
```bash
f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio "original_voice_ref/tien_bip/tien_bip2_trimmed.wav" \
--ref_text "anh ơi đến giờ phút này ý anh ạ..." \
--gen_text "YOUR_VIETNAMESE_TEXT_HERE" \
--speed 1.0 \
--vocoder_name vocos \
--vocab_file model/vocab.txt \
--ckpt_file model/model_last.pt \
```

**Programmatic Usage**:
```python
import sys
import subprocess
from pathlib import Path

# Option 1: Use existing CLI programmatically
def synthesize_speech_cli(text, ref_audio, ref_text, output_dir, speaker="tien_bip"):
    """
    Call F5-TTS-Vietnamese CLI from Python.
    """
    cmd = [
        "f5-tts_infer-cli",
        "--model", "F5TTS_Base",
        "--ckpt_file", "/home/psilab/F5-TTS-Vietnamese/model/model_last.pt",
        "--vocab_file", "/home/psilab/F5-TTS-Vietnamese/model/vocab.txt",
        "--ref_audio", ref_audio,
        "--ref_text", ref_text,
        "--gen_text", text,
        "--speed", "1.0",
        "--vocoder_name", "vocos",
        "--output_dir", output_dir,
        "--remove_silence"
    ]
    
    result = subprocess.run(cmd, cwd="/home/psilab/F5-TTS-Vietnamese", capture_output=True, text=True)
    return result

# Option 2: Use existing shell scripts
def synthesize_with_existing_script(text, speaker="tien_bip"):
    """
    Modify and run existing inference scripts.
    """
    script_path = f"/home/psilab/F5-TTS-Vietnamese/script_infer/infer_{speaker}.sh"
    # Modify script with new gen_text, then execute
    subprocess.run(["bash", script_path], cwd="/home/psilab/F5-TTS-Vietnamese")

# Option 3: Import F5-TTS modules directly
sys.path.append('/home/psilab/F5-TTS-Vietnamese')
from f5_tts.infer.utils_infer import infer_process

def synthesize_speech_direct(text, ref_audio, ref_text, output_path):
    """
    Direct Python API using F5-TTS modules.
    """
    result = infer_process(
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        model_obj=None,  # Will load default model
        vocoder_name="vocos",
        show_info=print,
        progress=print
    )
    return result

# Example usage
audio_path = synthesize_speech_cli(
    text="Xin chào các bạn, hôm nay chúng ta sẽ học về trí tuệ nhân tạo",
    ref_audio="/home/psilab/F5-TTS-Vietnamese/original_voice_ref/tien_bip/tien_bip2_trimmed.wav",
    ref_text="anh ơi đến giờ phút này ý anh ạ",
    output_dir="output/synthesized",
    speaker="tien_bip"
)
```

**Timing Considerations**:
- Match generated audio duration with original video segments
- Apply speed adjustments if transcription is longer/shorter than original
- Handle long text chunking (F5-TTS works best with ~200 characters per chunk)
- Preserve natural pauses and breathing in Vietnamese speech
- Consider Vietnamese syllable timing (typically slower than English)

---

## Step 4: Wav2Lip Synchronization Module

### Purpose
Synchronize lip movements of a target face with the synthesized audio.

### Implementation Status
**AVAILABLE** - Existing Wav2Lip installation at `/home/psilab/Wav2Lip/`

### Implementation Details

**Model**: Wav2Lip (pre-trained lip-sync model)

**Model Location**: `/home/psilab/Wav2Lip/`
- Checkpoint: `checkpoints/Wav2Lip-SD-NOGAN.pt`
- Inference script: `inference.py` (already implemented)
- Face detection: `face_detection/` module included

**Key Features**:
- ✅ Pre-trained Wav2Lip model ready to use
- ✅ Face detection and tracking built-in
- ✅ Lip-sync generation frame by frame
- ✅ Support for various video resolutions
- ✅ Quality preservation with multiple batch size options
- ✅ Existing inference script with comprehensive options

**Code Structure**:
```python
# scripts/wav2lip_sync.py

import subprocess
from pathlib import Path

def sync_lips_to_audio(
    face_video,
    audio_path,
    output_path,
    checkpoint_path="/home/psilab/Wav2Lip/checkpoints/Wav2Lip-SD-NOGAN.pt",
    face_det_batch_size=16,
    wav2lip_batch_size=128,
    resize_factor=1,
    fps=25,
    pads=[0, 10, 0, 0],
    nosmooth=False
):
    """
    Synchronize lip movements with audio using existing Wav2Lip installation.
    
    Args:
        face_video: Video containing target face
        audio_path: Generated audio to sync with
        output_path: Where to save synced video
        checkpoint_path: Path to Wav2Lip model checkpoint
        face_det_batch_size: Batch size for face detection (default: 16)
        wav2lip_batch_size: Batch size for Wav2Lip inference (default: 128)
        resize_factor: Resolution reduction factor (default: 1)
        fps: Frames per second (default: 25)
        pads: Padding [top, bottom, left, right] (default: [0, 10, 0, 0])
        nosmooth: Disable face detection smoothing (default: False)
    
    Returns:
        str: Path to lip-synced video
    """
    # Call existing Wav2Lip inference.py script
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path,
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        "--resize_factor", str(resize_factor),
        "--fps", str(fps),
        "--pads", *[str(p) for p in pads]
    ]
    
    if nosmooth:
        cmd.append("--nosmooth")
    
    result = subprocess.run(
        cmd,
        cwd="/home/psilab/Wav2Lip",
        capture_output=True,
        text=True
    )
    
    return output_path
```

**Existing Wav2Lip Installation**:
The Wav2Lip project is already set up at `/home/psilab/Wav2Lip/` with:
- ✅ `inference.py` - Main inference script
- ✅ `checkpoints/Wav2Lip-SD-NOGAN.pt` - Pre-trained model
- ✅ `face_detection/` - Face detection module
- ✅ `models/` - Model architecture definitions
- ✅ `audio.py` - Audio processing utilities

**CLI Usage** (using existing Wav2Lip script):
```bash
cd /home/psilab/Wav2Lip

# Basic usage
python inference.py \
  --checkpoint_path checkpoints/Wav2Lip-SD-NOGAN.pt \
  --face path/to/video.mp4 \
  --audio path/to/audio.wav \
  --outfile results/output.mp4

# With custom settings
python inference.py \
  --checkpoint_path checkpoints/Wav2Lip-SD-NOGAN.pt \
  --face path/to/video.mp4 \
  --audio path/to/audio.wav \
  --outfile results/output.mp4 \
  --face_det_batch_size 16 \
  --wav2lip_batch_size 128 \
  --resize_factor 1 \
  --fps 25 \
  --pads 0 10 0 0 \
  --nosmooth
```

**Programmatic Usage**:
```python
import subprocess
from pathlib import Path

def sync_lips_with_wav2lip(face_video, audio_path, output_path):
    """
    Use existing Wav2Lip installation for lip synchronization.
    """
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", "checkpoints/Wav2Lip-SD-NOGAN.pt",
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path,
        "--face_det_batch_size", "16",
        "--wav2lip_batch_size", "128"
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/home/psilab/Wav2Lip",
        capture_output=True,
        text=True,
        check=True
    )
    
    return output_path

# Example usage
synced_video = sync_lips_with_wav2lip(
    face_video="temp/downloads/VIDEO_ID/video.mp4",
    audio_path="temp/synthesized_audio.wav",
    output_path="temp/synced_video.mp4"
)
```

**Dependencies**:
Already installed in Wav2Lip directory. Check `requirements.txt`:
```bash
cd /home/psilab/Wav2Lip
cat requirements.txt
# Should include: torch, torchvision, opencv-python, librosa, scipy, etc.
```

**Face Source Options**:
1. **From YouTube video**: Use downloaded video directly (recommended)
2. **From static image**: Use a single photo (requires --static flag and --fps)
3. **From separate video**: Use pre-recorded face video

**Processing Steps** (handled by inference.py):
1. Face detection using built-in face_detection module
2. Face tracking across frames
3. Lip region extraction
4. Wav2Lip inference frame by frame
5. Face region blending back to original
6. Video compilation with synced audio

**Quality Settings**:
```python
# Default settings (can be adjusted)
CONFIG = {
    "checkpoint_path": "/home/psilab/Wav2Lip/checkpoints/Wav2Lip-SD-NOGAN.pt",
    "fps": 25,
    "face_det_batch_size": 16,  # Reduce if CUDA out of memory
    "wav2lip_batch_size": 128,  # Reduce if CUDA out of memory
    "resize_factor": 1,  # Increase to 2 or 4 for lower resolution (faster)
    "pads": [0, 10, 0, 0],  # [top, bottom, left, right] padding
    "nosmooth": False,  # Set True to disable face detection smoothing
}
```

**Performance Tips**:
- Use `--resize_factor 2` for 480p processing (faster, lower quality)
- Reduce `--wav2lip_batch_size` to 64 if running out of CUDA memory
- Use `--static True` for static image input (much faster)
- Add padding with `--pads` to ensure chin is included in detection

---

## Step 5: Video Composition Module

### Purpose
Overlay the lip-synced video in the top-right corner of the original YouTube video.

### Implementation Details

**Tool**: FFmpeg (powerful video processing library)

**Key Features**:
- Picture-in-picture overlay
- Position control (top-right corner)
- Size adjustment
- Border/shadow effects (optional)
- Audio mixing

**Code Structure**:
```python
# scripts/compose_video.py

import subprocess
from pathlib import Path

def compose_picture_in_picture(
    main_video,
    overlay_video,
    output_path,
    position="top-right",
    overlay_size=(480, 270),  # 1/4 of 1080p
    margin=(20, 20)
):
    """
    Overlay synced video on original video using FFmpeg.
    
    Args:
        main_video: Original YouTube video
        overlay_video: Lip-synced video to overlay
        output_path: Final output video path
        position: Overlay position (top-right, top-left, etc.)
        overlay_size: Size of overlay in pixels
        margin: Margin from edges in pixels
    
    Returns:
        str: Path to final composed video
    """
    # Build FFmpeg command
    # Scale overlay video
    # Position overlay
    # Mix audio tracks
    # Encode final video
```

**FFmpeg Command Structure**:
```bash
ffmpeg -i main_video.mp4 -i overlay_video.mp4 \
  -filter_complex "[1:v]scale=480:270[overlay]; \
                   [0:v][overlay]overlay=W-w-20:20[outv]; \
                   [0:a][1:a]amix=inputs=2:duration=shortest[outa]" \
  -map "[outv]" -map "[outa]" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 192k \
  output.mp4
```

**Position Calculations**:
```python
POSITIONS = {
    "top-right": "W-w-{margin_x}:{margin_y}",
    "top-left": "{margin_x}:{margin_y}",
    "bottom-right": "W-w-{margin_x}:H-h-{margin_y}",
    "bottom-left": "{margin_x}:H-h-{margin_y}",
}
```

**Enhancement Options**:
- Add border around overlay
- Add drop shadow
- Adjust opacity
- Add rounded corners
- Fade in/out effects

**Dependencies**:
```bash
# Option 1: System-level installation (recommended for full features)
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS

# Option 2: Install via pip in virtual environment (portable)
pip install imageio-ffmpeg  # Includes FFmpeg binaries
pip install ffmpeg-python   # Python wrapper for FFmpeg commands
```

---

## Step 6: Pipeline Orchestration

### Purpose
Tie all modules together into a single automated pipeline.

### Implementation Details

**Main Pipeline Script**:
```python
# pipeline_vidlipsyncvoice.py

import argparse
from pathlib import Path
import json

# Import existing modules
from scripts.download_youtube_video import download_youtube_video
from scripts.transcribe_phowhisper import transcribe_with_timestamps
from scripts.synthesize_voice import synthesize_speech
from scripts.wav2lip_sync import sync_lips_to_audio
from scripts.compose_video import compose_picture_in_picture

def run_pipeline(
    youtube_url,
    target_speaker,
    ref_audio,
    ref_text,
    output_dir="output/pipeline",
    intermediate_cleanup=True
):
    """
    Run complete VidLipSyncVoice pipeline.
    
    Args:
        youtube_url: URL of YouTube video to process
        target_speaker: Name of target speaker for voice synthesis
        ref_audio: Path to reference audio for target speaker
        ref_text: Transcription of reference audio
        output_dir: Directory for final output
        intermediate_cleanup: Whether to delete intermediate files
    
    Returns:
        str: Path to final composed video
    """
    
    print("=" * 60)
    print("VidLipSyncVoice Pipeline")
    print("=" * 60)
    
    # Step 1: Download YouTube video
    print("\n[1/6] Downloading YouTube video...")
    download_result = download_youtube_video(youtube_url, "temp/youtube")
    video_path = download_result['video']
    audio_path = download_result['audio']
    
    # Step 2: Transcribe with timestamps
    print("\n[2/6] Transcribing audio with PhoWhisper...")
    transcription = transcribe_with_timestamps(audio_path)
    full_text = transcription['text']
    
    # Save transcription
    trans_path = Path("temp") / "transcription.json"
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)
    
    # Step 3: Synthesize speech with F5-TTS
    print("\n[3/6] Synthesizing speech with F5-TTS...")
    synth_audio_path = Path("temp") / "synthesized_audio.wav"
    
    # Use synthesize_voice.py wrapper
    synthesize_speech(
        text=full_text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        output_path=str(synth_audio_path)
    )
    
    # Step 4: Prepare face video source
    print("\n[4/6] Preparing face video...")
    # Use YouTube video itself
    face_video = video_path
    
    # Step 5: Sync lips with Wav2Lip
    print("\n[5/6] Synchronizing lips with Wav2Lip...")
    synced_video_path = Path("temp") / "synced_video.mp4"
    
    # Use wav2lip_sync.py wrapper
    sync_lips_to_audio(
        face_video=str(face_video),
        audio_path=str(synth_audio_path),
        output_path=str(synced_video_path)
    )
    
    # Step 6: Compose final video
    print("\n[6/6] Composing final video...")
    final_output = Path(output_dir) / f"final_{target_speaker}.mp4"
    final_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Use compose_video.py wrapper
    compose_picture_in_picture(
        main_video=str(video_path),
        overlay_video=str(synced_video_path),
        output_path=str(final_output),
        position="top-right"
    )
    
    # Cleanup intermediate files
    if intermediate_cleanup:
        print("\nCleaning up intermediate files...")
        # Remove temp files but keep final output
    
    print(f"\n✅ Pipeline complete! Output saved to: {final_output}")
    return str(final_output)

def main():
    parser = argparse.ArgumentParser(description="VidLipSyncVoice Pipeline")
    parser.add_argument("youtube_url", help="YouTube video URL")
    parser.add_argument("--speaker", default="tien_bip", help="Target speaker name")
    parser.add_argument("--ref_audio", help="Reference audio path")
    parser.add_argument("--ref_text", help="Reference audio transcription")
    parser.add_argument("--output_dir", default="output/pipeline", help="Output directory")
    parser.add_argument("--keep_temp", action="store_true", help="Keep intermediate files")
    
    args = parser.parse_args()
    
    # Use default reference if not provided
    if not args.ref_audio:
        # Use F5-TTS-Vietnamese reference speakers
        args.ref_audio = f"/home/psilab/F5-TTS-Vietnamese/original_voice_ref/{args.speaker}/{args.speaker}_trimmed.wav"
    
    if not args.ref_text:
        # You may want to provide a default ref_text or load it from a file
        args.ref_text = "Đây là giọng nói mẫu"
    
    run_pipeline(
        youtube_url=args.youtube_url,
        target_speaker=args.speaker,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_dir=args.output_dir,
        intermediate_cleanup=not args.keep_temp
    )

if __name__ == "__main__":
    main()
```

**Key Integration Points**:
1. **Step 1-2**: Direct Python imports - `download_youtube_video()`, `transcribe_with_timestamps()`
2. **Step 3**: Python wrapper - `synthesize_speech()` from `scripts/synthesize_voice.py`
3. **Step 4**: Python wrapper - `sync_lips_to_audio()` from `scripts/wav2lip_sync.py`
4. **Step 5**: Python wrapper - `compose_picture_in_picture()` from `scripts/compose_video.py`

**Dependencies**:
- Python scripts in `VidLipSyncVoice/scripts/`: 
  - ✅ `download_youtube_video.py` (implemented)
  - ✅ `transcribe_phowhisper.py` (implemented)
  - ⏳ `synthesize_voice.py` (TODO - wraps F5-TTS-Vietnamese CLI)
  - ⏳ `wav2lip_sync.py` (TODO - wraps Wav2Lip inference.py)
  - ⏳ `compose_video.py` (TODO - wraps FFmpeg commands)
- External tools: F5-TTS-Vietnamese, Wav2Lip, FFmpeg (called by wrapper scripts)
- Python packages: `imageio-ffmpeg`, `ffmpeg-python` (or system FFmpeg)

**Implementation Notes**:
- Each wrapper script (steps 3-5) handles subprocess calls internally
- Wrapper scripts provide clean Python API for pipeline orchestration
- Error handling and path validation done within wrapper functions
- Makes the main pipeline script cleaner and more maintainable

---

## Installation & Setup

### System Requirements
- Python 3.8+
- FFmpeg
- CUDA-capable GPU (recommended for Wav2Lip and F5-TTS)
- 8GB+ RAM
- 10GB+ disk space for models

### Installation Steps

```bash
# 1. Install Python dependencies
pip install -r requirements_pipeline.txt

# 2. Install FFmpeg
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS

# 3. Download Wav2Lip repository and models
cd /home/psilab/F5-TTS-Vietnamese
git clone https://github.com/Rudrabha/Wav2Lip.git models/Wav2Lip

# Download pretrained models
mkdir -p models/wav2lip_models
cd models/wav2lip_models
wget "https://github.com/Rudrabha/Wav2Lip/releases/download/models/wav2lip_gan.pth"
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

# 4. Install PhoWhisper (or use standard Whisper)
pip install git+https://github.com/openai/whisper.git
# OR for PhoWhisper if available
pip install transformers

# 5. Verify F5-TTS model is present
ls model/model_last.pt model/vocab.txt
```

### Dependencies File

Create `requirements_pipeline.txt`:
```
yt-dlp>=2023.10.13
opencv-python>=4.8.0
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
librosa>=0.10.0
soundfile>=0.12.0
openai-whisper>=20231117
transformers>=4.35.0
accelerate>=0.24.0
face-recognition>=1.3.0
scipy>=1.10.0
numba>=0.58.0
```

---

## Usage Examples

### Basic Usage
```bash
python pipeline_vidlipsyncvoice.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --speaker tien_bip \
  --output_dir output/pipeline
```

### Advanced Usage with Custom Reference
```bash
python pipeline_vidlipsyncvoice.py "https://www.youtube.com/watch?v=VIDEO_ID" \
  --speaker custom_speaker \
  --ref_audio "path/to/reference.wav" \
  --ref_text "transcription of reference audio" \
  --output_dir output/custom \
  --keep_temp
```

### Batch Processing
```bash
# Create a batch script
cat > batch_process.sh << 'EOF'
#!/bin/bash
while IFS=',' read -r url speaker; do
  python pipeline_vidlipsyncvoice.py "$url" --speaker "$speaker"
done < video_list.csv
EOF

# video_list.csv format:
# https://youtube.com/watch?v=ID1,tien_bip
# https://youtube.com/watch?v=ID2,kha_banh
```

---

## File Organization

### Directory Structure After Implementation

```
VidLipSyncVoice/                         # Current project directory
├── scripts/                             # Utility modules
│   ├── download_youtube_video.py        # ✅ Step 1: YouTube download (IMPLEMENTED)
│   └── transcribe_phowhisper.py         # ✅ Step 2: Transcription (IMPLEMENTED)
│
├── temp/                                # Temporary processing files
│   └── downloads/                       # Downloaded YouTube videos
│       └── VIDEO_ID_Title/
│           ├── video.mp4                # Original video
│           ├── audio.wav                # Extracted audio
│           ├── thumbnail.jpg            # Video thumbnail
│           ├── metadata.json            # Video metadata
│           └── audio_transcription.json # Transcription output
│
├── venv/                                # Virtual environment
├── requirements.txt                     # Python dependencies
├── PIPELINE_IMPLEMENTATION_PLAN.md      # This document
├── PHOWHISPER_USAGE.md                 # PhoWhisper usage guide
└── README.md                            # Project documentation

F5-TTS-Vietnamese/                       # Referenced F5-TTS project
├── model/
│   ├── model_last.pt                    # F5-TTS model (for Step 3)
│   └── vocab.txt
└── original_voice_ref/                  # Reference speakers (for Step 3)
    ├── tien_bip/
    ├── kha_banh/
    └── ...

models--vinai--PhoWhisper-medium/        # PhoWhisper model location
└── snapshots/
    └── 55a7e3eb6c906de891f8f06a107754427dd3be79/
        ├── config.json
        ├── pytorch_model.bin
        └── ...

Wav2Lip/                                 # Wav2Lip project (for Step 4)
└── checkpoints/
    └── Wav2Lip-SD-NOGAN.pt
```

### Implemented Steps Status

- ✅ **Step 1**: YouTube Download (`scripts/download_youtube_video.py`)
- ✅ **Step 2**: PhoWhisper Transcription (`scripts/transcribe_phowhisper.py`)
- ⏳ **Step 3**: F5-TTS Voice Synthesis (TODO)
- ⏳ **Step 4**: Wav2Lip Lip Sync (TODO)
- ⏳ **Step 5**: Video Composition (TODO)
- ⏳ **Step 6**: Pipeline Orchestration (TODO)

---

## Testing Strategy

### Unit Tests for Each Module

```python
# tests/test_pipeline_modules.py

def test_youtube_download():
    """Test YouTube video download functionality"""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = download_youtube_video(url, "temp/test")
    assert Path(result['video']).exists()
    assert Path(result['audio']).exists()

def test_transcription():
    """Test PhoWhisper transcription"""
    audio = "tests/sample_audio.wav"
    result = transcribe_with_timestamps(audio)
    assert 'text' in result
    assert 'segments' in result
    assert len(result['segments']) > 0

def test_voice_synthesis():
    """Test F5-TTS synthesis"""
    text = "xin chào"
    ref_audio = "original_voice_ref/tien_bip/tien_bip2_trimmed.wav"
    output = "temp/test_synth.wav"
    result = synthesize_speech(text, ref_audio, "", output)
    assert Path(result).exists()

# ... more tests
```

### Integration Test

```python
def test_full_pipeline():
    """Test complete pipeline end-to-end"""
    # Use a short test video
    test_url = "SHORT_TEST_VIDEO_URL"
    output = run_pipeline(
        youtube_url=test_url,
        target_speaker="tien_bip",
        ref_audio="original_voice_ref/tien_bip/tien_bip2_trimmed.wav",
        ref_text="test reference",
        output_dir="temp/test_output"
    )
    assert Path(output).exists()
    # Verify video properties (duration, resolution, etc.)
```

---

## Performance Optimization

### Processing Time Estimates
- YouTube download: 10-30 seconds (depends on video length)
- Transcription: 0.1x - 0.3x real-time (10min video = 1-3min)
- Voice synthesis: 0.2x - 0.5x real-time
- Wav2Lip: 1x - 3x real-time (slowest step)
- Video composition: 0.1x - 0.2x real-time

**Total for 5-minute video: ~5-15 minutes**

### Optimization Strategies

1. **Parallel Processing**:
   - Process audio and video tracks separately
   - Use multi-threading for frame processing

2. **GPU Acceleration**:
   - Use CUDA for Wav2Lip inference
   - Batch processing for F5-TTS

3. **Quality vs Speed Trade-offs**:
   - Lower resolution for faster Wav2Lip
   - Reduce audio sample rate if acceptable
   - Use faster Whisper models (base vs large)

4. **Caching**:
   - Cache downloaded videos
   - Cache transcriptions
   - Reuse models between runs

---

## Error Handling & Edge Cases

### Common Issues and Solutions

1. **YouTube Download Failures**:
   - Retry with exponential backoff
   - Handle geo-restrictions
   - Support alternative download methods

2. **Transcription Accuracy**:
   - Provide manual transcription override
   - Support multiple language models
   - Confidence score filtering

3. **Face Detection Failures**:
   - Fallback to alternative face detection
   - Manual face ROI specification
   - Use alternative video source

4. **Audio/Video Sync Issues**:
   - Add padding/trimming logic
   - Speed adjustment
   - Manual timestamp correction

5. **Memory Issues**:
   - Process videos in chunks
   - Reduce batch sizes
   - Clear cache between steps

---

## Future Enhancements

### Phase 2 Features

1. **Multi-Speaker Support**:
   - Detect and handle multiple speakers
   - Assign different voices to different speakers

2. **Real-time Processing**:
   - Stream processing for live videos
   - Reduced latency pipeline

3. **Advanced Composition**:
   - Multiple overlay positions
   - Custom layouts
   - Transition effects

4. **Quality Improvements**:
   - Super-resolution for face region
   - Better audio quality preservation
   - Smoother lip-sync transitions

5. **Web Interface**:
   - Gradio UI for easy access
   - Progress tracking
   - Result gallery

6. **Cloud Deployment**:
   - API service
   - Queue-based processing
   - Distributed computing

---

## Troubleshooting Guide

### Common Problems

**Problem**: "FFmpeg not found"
```bash
# Solution: Install FFmpeg
sudo apt install ffmpeg
# Verify: ffmpeg -version
```

**Problem**: "CUDA out of memory"
```python
# Solution: Reduce batch sizes
CONFIG = {
    "wav2lip_batch_size": 64,  # Reduce from 128
    "face_det_batch_size": 8,   # Reduce from 16
}
```

**Problem**: "Poor lip-sync quality"
```
# Solutions:
1. Use higher quality face video
2. Ensure face is clearly visible
3. Use static camera angle
4. Increase Wav2Lip model quality (use GAN version)
```

**Problem**: "Vietnamese transcription errors"
```python
# Solution: Use larger Whisper model
transcribe_with_timestamps(audio, model_name="large-v3")
# Or manually provide transcription
```

---

## Implementation Timeline

### Estimated Development Schedule

**Week 1**: Core Infrastructure
- Day 1-2: YouTube download module
- Day 3-4: PhoWhisper transcription module
- Day 5: Testing and integration

**Week 2**: Voice Processing
- Day 1-3: F5-TTS integration and voice synthesis
- Day 4-5: Wav2Lip setup and lip-sync module

**Week 3**: Video Processing & Integration
- Day 1-2: Video composition module
- Day 3-4: Main pipeline orchestration
- Day 5: Integration testing

**Week 4**: Polish & Documentation
- Day 1-2: Error handling and edge cases
- Day 3: Performance optimization
- Day 4-5: Documentation and examples

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building the VidLipSyncVoice pipeline. The modular design allows for:

- **Flexibility**: Each module can be developed and tested independently
- **Maintainability**: Clear separation of concerns
- **Extensibility**: Easy to add new features or swap components
- **Reusability**: Modules can be used in other projects

**Next Steps**:
1. Review and approve this plan
2. Set up development environment
3. Begin implementation with Step 1 (YouTube download)
4. Iterate and test each module
5. Integrate into complete pipeline
6. Deploy and document

The existing F5-TTS Vietnamese infrastructure provides a strong foundation, and the additional components (Wav2Lip, FFmpeg composition) are well-established technologies with good documentation and community support.
