from fastapi import HTTPException
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import llm
from prompts.prompts import system_prompt
from services.redis_service import RedisSessionManager, get_redis_history
from services.tts_service import text_to_speech
from services.transcription_service import transcribe_audio

class AIInterviewer:
    def __init__(self):
        self.llm = llm
        self.session_manager = RedisSessionManager()
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        runnable = prompt | self.llm
        return RunnableWithMessageHistory(
            runnable, 
            get_redis_history, 
            input_messages_key="input", 
            history_messages_key="history"
        )

    def get_profile(self, session_id: str) -> dict:
        profile = self.session_manager.get_profile(session_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Invalid session ID")
        return profile

    def text_to_speech(self, text: str) -> str:
        return text_to_speech(text)

    def transcribe_audio(self, filename: str) -> str:
        return transcribe_audio(filename)