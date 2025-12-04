#!/usr/bin/env python3
"""
Simple FastAPI for Audio Transcription
Input: WAV file
Output: Transcribed text
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from pathlib import Path
import sys
import json
import re

# Add parent directory to path to import transcription module
sys.path.append(str(Path(__file__).parent.parent))

try:
    from scripts.transcribe_phowhisper import transcribe_with_timestamps
except ImportError:
    print("Warning: Could not import transcribe_phowhisper, using fallback")
    transcribe_with_timestamps = None

app = FastAPI(
    title="Audio Transcription API",
    description="Simple API to transcribe WAV audio files to text",
    version="1.0.0",
    docs_url="/docs"
)

# Configure model path (adjust as needed)
MODEL_PATH = "/home/psilab/TRANSCRIBE-AUDIO-TO-TEXT-WHISPER/model/snapshots/55a7e3eb6c906de891f8f06a107754427dd3be79"


def clean_unk_tokens(text):
    """Remove 'unk' tokens and excessive punctuation from transcription."""
    if not text:
        return text
    
    # Remove standalone 'unk' tokens (with or without punctuation)
    text = re.sub(r'\bunk\b\.?\s*', '', text, flags=re.IGNORECASE)
    
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple consecutive periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove trailing/leading whitespace
    text = text.strip()
    
    return text


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Transcription API",
        "endpoints": {
            "/transcribe": "POST - Upload WAV file for transcription",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "vi",
    add_punctuation: bool = True
):
    """
    Transcribe audio file to text.
    
    Args:
        file: WAV audio file
        language: Language code (default: "vi" for Vietnamese)
        add_punctuation: Whether to restore punctuation (default: True)
    
    Returns:
        JSON with transcribed text and metadata
    """
    # Validate file extension
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported. Please upload a .wav file."
        )
    
    # Check if transcription function is available
    if transcribe_with_timestamps is None:
        raise HTTPException(
            status_code=500,
            detail="Transcription module not available. Please check dependencies."
        )
    
    # Create temporary file to save uploaded audio
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Write uploaded file to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Perform transcription
        result = transcribe_with_timestamps(
            audio_path=temp_path,
            model_path=MODEL_PATH,
            language=language,
            verbose=False,
            add_punctuation=add_punctuation
        )
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Clean up unk tokens from the text
        cleaned_text = clean_unk_tokens(result['text'])
        cleaned_text_no_punct = clean_unk_tokens(result.get('text_no_punctuation', ''))
        
        # Return results (FastAPI will auto-serialize dict with proper UTF-8)
        return {
            "success": True,
            "filename": file.filename,
            "text": cleaned_text,
            "text_no_punctuation": cleaned_text_no_punct,
            "language": result['language'],
            "duration": result['duration'],
            "segments_count": len(result['segments']),
            "punctuation_restored": result.get('punctuation_restored', False)
        }
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {str(e)}"
        )
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
