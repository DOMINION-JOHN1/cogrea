from pydantic import BaseModel
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from langchain.schema.runnable import RunnablePassthrough
from langgraph.graph.message import add_messages

# Enhanced State Schema for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    history: Annotated[str, RunnablePassthrough()]
    career_summary: Annotated[str, RunnablePassthrough()]
    learning_roadmap: Annotated[str, RunnablePassthrough()]
    personalized_schedule: Annotated[str, RunnablePassthrough()]

# Pydantic Models for API validation
class UserProfile(BaseModel):
    firstName: str
    lastName: str
    country: str

class InterviewRequest(BaseModel):
    profile: dict

class TextResponse(BaseModel):
    session_id: str
    text: str