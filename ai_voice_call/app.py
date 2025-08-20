import os
import io
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
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

# User profile model
class UserProfile(BaseModel):
    firstName: str
    lastName: str
    country: str

def create_personalized_system_instruction(profile: UserProfile) -> str:
    """Create a personalized system instruction based on user profile."""
    return f"""You are Cogrea, a helpful AI assistant specializing in career guidance and personal development. 

You are speaking with {profile.firstName} {profile.lastName} from {profile.country}. 

Start the conversation by warmly greeting them by name and acknowledging their location. Be culturally aware and respectful of their background from {profile.country}. 

Your role is to:
- Provide personalized career advice and exploration guidance
- Offer learning path recommendations tailored to their context
- Help with study and work scheduling
- Answer questions about various topics relevant to their career journey

Be professional, supportive, warm, and provide actionable advice. Use their name naturally in conversation and consider their cultural context from {profile.country} when giving advice. If you need more information to help effectively, don't hesitate to ask clarifying questions.

Begin by introducing yourself and asking how you can help {profile.firstName} today with their career journey."""

def create_config(profile: Optional[UserProfile] = None) -> types.LiveConnectConfig:
    """Create LiveConnectConfig with personalized or default system instruction."""
    if profile:
        system_instruction = create_personalized_system_instruction(profile)
    else:
        system_instruction = "You are Cogrea, a helpful assistant focused on career guidance. Answer in a friendly tone."
    
    return types.LiveConnectConfig(
        response_modalities=[Modality.AUDIO],
        system_instruction=system_instruction,
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

# Store user profiles temporarily (in production, use a database)
user_sessions = {}

@app.post("/api/start_voice_session")
async def start_voice_session(profile: UserProfile):
    """Initialize a voice session with user profile."""
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = profile
    
    return {
        "session_id": session_id,
        "message": f"Voice session created for {profile.firstName} {profile.lastName}",
        "websocket_url": f"/ws/audio/{session_id}"
    }

@app.websocket("/ws/audio/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"WebSocket connection established for session: {session_id}")
    
    # Get user profile from session
    profile = user_sessions.get(session_id)
    if profile:
        print(f"Found profile for {profile.firstName} {profile.lastName} from {profile.country}")
    
    config = create_config(profile)
    
    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-preview-native-audio-dialog", config=config
        ) as session:
            # Track connection state
            connected = True
            
            async def send_audio():
                nonlocal connected
                try:
                    while connected:
                        data = await websocket.receive_bytes()
                        if data:  # Validate data exists
                            await session.send_realtime_input(
                                audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                            )
                except WebSocketDisconnect:
                    connected = False
                    print("Client disconnected during audio send")
                except Exception as e:
                    connected = False
                    print(f"Error in send_audio: {e}")

            async def receive_audio():
                nonlocal connected
                try:
                    async for response in session.receive():
                        if not connected:
                            break
                        if response.data is not None:
                            await websocket.send_bytes(response.data)
                except Exception as e:
                    connected = False
                    print(f"Error in receive_audio: {e}")

            await asyncio.gather(send_audio(), receive_audio(), return_exceptions=True)
            
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass  # WebSocket might already be closed
    finally:
        # Clean up session
        if session_id in user_sessions:
            del user_sessions[session_id]
            print(f"Cleaned up session: {session_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)