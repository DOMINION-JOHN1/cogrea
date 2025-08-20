import asyncio
import io
import os
import wave
from pathlib import Path
from dotenv import load_dotenv
import soundfile as sf
import librosa
from google import genai
from google.genai import types
from google.genai.types import (
    Modality, RealtimeInputConfig, AutomaticActivityDetection, 
    StartSensitivity, EndSensitivity, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
)

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)



config = types.LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction= "You are a helpful assistant and answer in a friendly tone.",
    realtime_input_config=RealtimeInputConfig(
        automatic_activity_detection=AutomaticActivityDetection(
            disabled=False,
            start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_LOW,
            silence_duration_ms=800
        )
    ),
    #proactive_audio=True,
    context_window_compression=types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow()
    ),
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
        ),
        language_code="en-US"
    )
)


async def main():
    async with client.aio.live.connect(model="gemini-2.5-flash-preview-native-audio-dialog", config=config) as session:
        buffer = io.BytesIO()
        y, sr = librosa.load("fifth reply.wav", sr=16000)
        sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
        buffer.seek(0)
        audio_bytes = buffer.read()

        # If already in correct format, you can use this:
        # audio_bytes = Path("sample.pcm").read_bytes()

        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )

        wf = wave.open("audio.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)  # Output is 24kHz

        async for response in session.receive():
            if response.data is not None:
                wf.writeframes(response.data)

            # Un-comment this code to print audio data info
            # if response.server_content.model_turn is not None:
            #      print(response.server_content.model_turn.parts[0].inline_data.mime_type)

        wf.close()

if __name__ == "__main__":
    asyncio.run(main())