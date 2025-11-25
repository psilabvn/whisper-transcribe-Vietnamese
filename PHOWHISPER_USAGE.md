# PhoWhisper Transcription - Quick Start

## Your PhoWhisper Model Location
```
/home/psilab/models--vinai--PhoWhisper-medium/snapshots/55a7e3eb6c906de891f8f06a107754427dd3be79/
```

## Installation

```bash
cd /home/psilab/VidLipSyncVoice
source venv/bin/activate
pip install transformers torch accelerate
```

## Usage Options

### Option 1: Use Your Existing PhoWhisper Model (Recommended)

The script automatically detects your PhoWhisper model:
```bash
python scripts/transcribe_phowhisper.py temp/downloads/*/audio.wav --model phowhisper
```

Or specify the path explicitly:
```bash
python scripts/transcribe_phowhisper.py temp/downloads/*/audio.wav \
  --model-path /home/psilab/models--vinai--PhoWhisper-medium/snapshots/55a7e3eb6c906de891f8f06a107754427dd3be79
```

### Option 2: Use OpenAI Whisper (Alternative)

If you have `openai-whisper` installed:
```bash
# Small model (fast)
python scripts/transcribe_phowhisper.py temp/downloads/*/audio.wav --model base

# Medium model (better accuracy)
python scripts/transcribe_phowhisper.py temp/downloads/*/audio.wav --model medium
```

## Example: Transcribe Your Downloaded Video

```bash
# Find your downloaded audio file
ls temp/downloads/*/audio.wav

# Transcribe with PhoWhisper
python scripts/transcribe_phowhisper.py temp/downloads/EQ1JUda3tYk_*/audio.wav --model phowhisper

# Output will be saved as: temp/downloads/EQ1JUda3tYk_*/audio_transcription.json
```

## Show Detailed Output

```bash
# Show first 5 segments with word-level timestamps
python scripts/transcribe_phowhisper.py temp/downloads/*/audio.wav \
  --model phowhisper \
  --show-segments 5
```

## Custom Output Location

```bash
python scripts/transcribe_phowhisper.py temp/downloads/*/audio.wav \
  --model phowhisper \
  --output my_transcription.json
```

## No Need to Move the Model!

Your PhoWhisper model can stay where it is. The script will load it from:
`/home/psilab/models--vinai--PhoWhisper-medium/`

This avoids duplicating the model files and saves disk space.
