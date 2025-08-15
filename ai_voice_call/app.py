import os
import io
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from google.genai.types import (
    Modality, RealtimeInputConfig, AutomaticActivityDetection,
    StartSensitivity, EndSensitivity, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
)
from google.genai import types
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

config = types.LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction="You are a helpful assistant and answer in a friendly tone.",
    realtime_input_config=RealtimeInputConfig(
        automatic_activity_detection=AutomaticActivityDetection(
            disabled=False,
            start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_LOW,
            silence_duration_ms=800
        )
    ),
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

app = FastAPI()

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-preview-native-audio-dialog", config=config
        ) as session:
            async def send_audio():
                while True:
                    data = await websocket.receive_bytes()
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )

            async def receive_audio():
                async for response in session.receive():
                    if response.data is not None:
                        await websocket.send_bytes(response.data)

            await asyncio.gather(send_audio(), receive_audio())
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)