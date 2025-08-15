import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from core.interviewer import AIInterviewer
from core.workflow import workflow
from models.schemas import UserProfile, InterviewRequest, TextResponse
from services.redis_service import WindowsRedisChatMessageHistory
from langchain.schema import HumanMessage
# Initialize the workflow
from core.workflow import workflow, process_message




# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router and AIInterviewer
router = APIRouter()
interviewer = AIInterviewer()

@router.post("/start_interview")
async def start_interview(request: InterviewRequest):
    try:
        logger.info(f"Received profile data: {request.profile}")
        profile = UserProfile(**request.profile).model_dump()
        session_id = interviewer.session_manager.create_session(profile)
        
        initial_prompt = f"My name is {profile['firstName']}."
        
        response = process_message(
            session_id=session_id,  # Unique ID for the conversation
            user_message=initial_prompt
)

        
        #audio_path = interviewer.text_to_speech(response.content)
        
        return JSONResponse({
            "session_id": session_id,
            "text": response,
            #"audio_url": f"/audio/{audio_path}"
        })
    except Exception as e:
        logger.error(f"Start interview error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/process_text_response")
async def process_text_response(response: TextResponse):
    try:
        # Process a user message
        response = process_message(
            session_id=response.session_id,  # Unique ID for the conversation
            user_message=response.text
)
        #audio_path = interviewer.text_to_speech(ai_response.content)
        return JSONResponse({
            "text": response,
            #"audio_url": f"/audio/{audio_path}"
        })
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_voice_response")
async def process_voice_response( 
    session_id: str=Form(...),
    audio: UploadFile = File(...),):
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_file:
            temp_file.write(await audio.read())
            temp_file.flush()
            user_input = interviewer.transcribe_audio(temp_file.name)
        
        ai_response = interviewer.chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        #audio_path = interviewer.text_to_speech(ai_response.content)
        
        return JSONResponse({
            "text": ai_response.content,
            #"audio_url": f"/audio/{audio_path}"
        })
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate_interview/{session_id}")
async def evaluate_interview(session_id: str):
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")
    try:
        history_extractor = WindowsRedisChatMessageHistory(session_id=session_id)
        chat_history = history_extractor.messages
        history = "\n".join([
            f"{'Candidate' if isinstance(msg, HumanMessage) else 'Interviewer'}: {msg.content}"
            for msg in chat_history
        ]) or "No conversation history"
        
        return history
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail="Evaluation failed")


@router.delete("/end_interview/{session_id}")
async def end_interview(session_id: str):
    try:
        interviewer.session_manager.delete_session(session_id)
        return {"status": "session closed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}