import logging
from fastapi import HTTPException
from config import groq_client

logger = logging.getLogger(__name__)

def transcribe_audio(filename: str) -> str:
    try:
        with open(filename, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3",
                response_format="text"
            )
        return result
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Audio transcription failed")