import uuid
import logging
from fastapi import HTTPException
from elevenlabs import VoiceSettings
from config import elevenlabs_client

logger = logging.getLogger(__name__)

def text_to_speech(text: str) -> str:
    try:
        response = elevenlabs_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.47,
                similarity_boost=0.21,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )
        
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="temp_audio_", dir=None)
        try:
            for chunk in response:
                if chunk:
                    temp_file.write(chunk)
            temp_file.close()
            return temp_file.name
        except Exception as file_error:
            temp_file.close()
            import os
            os.unlink(temp_file.name)
            raise file_error
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail="Audio generation failed")