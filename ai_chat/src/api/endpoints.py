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
        
        ai_response = process_message(
            session_id=session_id,  # Unique ID for the conversation
            user_message=initial_prompt
        )

        #audio_path = interviewer.text_to_speech(ai_response.content)
        
        return JSONResponse({
            "session_id": session_id,
            "text": ai_response,
            #"audio_url": f"/audio/{audio_path}"
        })
    except Exception as e:
        logger.error(f"Start interview error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/process_text_response")
async def process_text_response(request: TextResponse):
    try:
        logger.info(f"Received request: {request}")
        logger.info(f"Session ID: {request.session_id}")
        logger.info(f"Text: {request.text}")
        
        # Validate inputs
        if not request.session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        if not request.text:
            raise HTTPException(status_code=400, detail="text is required")
        
        logger.info("About to call process_message...")
        
        # Process the user message - use different variable name to avoid conflict
        ai_response = process_message(
            session_id=request.session_id,
            user_message=request.text
        )
        
        logger.info(f"Generated response: {ai_response}")
        logger.info(f"Response type: {type(ai_response)}")
        
        # Ensure the response is a string
        if ai_response is None:
            ai_response = "I apologize, I couldn't generate a response. Please try again."
        elif not isinstance(ai_response, str):
            ai_response = str(ai_response)
        
        #audio_path = interviewer.text_to_speech(ai_response)
        return JSONResponse({
            "text": ai_response,
            #"audio_url": f"/audio/{audio_path}"
        })
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Text processing error: {e}", exc_info=True)
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
        
        # Use the workflow instead of the interviewer chain
        ai_response = process_message(
            session_id=session_id,
            user_message=user_input
        )
        
        #audio_path = interviewer.text_to_speech(ai_response)
        
        return JSONResponse({
            "text": ai_response,
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
        
        return {"conversation_history": history}
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