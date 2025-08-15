from typing import Dict, List, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from config import REDIS_HOST, REDIS_PASSWORD,llm, tavily_tool , REDIS_URL

# Define the state for our conversation
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda a, b: a + b]

# System prompt that defines the AI's role and capabilities
SYSTEM_PROMPT = """You are Cogrea, a helpful AI assistant. You are knowledgeable about career guidance, 
learning paths, and scheduling. You can help users with:
- Career advice and exploration
- Learning path recommendations
- Study and work scheduling
- Answering questions about various topics

Be professional, supportive, and provide actionable advice. If you need more information to help effectively, 
don't hesitate to ask clarifying questions."""

def get_conversation_chain():
    # Initialize the prompt with system message and chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # Create the chain
    chain = prompt | llm
    
    return chain

def call_model(state: AgentState, config: RunnableConfig):
    # Get the conversation chain
    chain = get_conversation_chain()
    
    # Invoke the model with the current conversation history
    response = chain.invoke({"messages": state["messages"]}, config)
    
    # Return the response to be added to the conversation
    return {"messages": [response]}

def build_workflow():
    # Create a new workflow
    builder = StateGraph(AgentState)
    
    # Add the model node
    builder.add_node("call_model", call_model)
    builder.add_node("research_assistant", ToolNode([tavily_tool]))

    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )

    builder.add_edge("tools", "chatbot")
    builder.add_edge(START,"call_model")
   
    
    # Set up Redis checkpointing
    with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        checkpointer.setup()
        # Compile the graph with the Redis checkpointer
        return builder.compile(checkpointer=checkpointer)

# Create the workflow instance
workflow = build_workflow()

def process_message(session_id: str, user_message: str) -> str:
    """Process a user message and return the AI's response."""
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    # Create a human message from the user input
    message = HumanMessage(content=user_message)
    
    # Invoke the workflow to get the response
    result = workflow.invoke(
        {"messages": [message]},
        config
    )
    
    # Get the last message (AI's response)
    if result["messages"] and hasattr(result["messages"][-1], 'content'):
        return result["messages"][-1].content
    
    return "I'm sorry, I couldn't generate a response. Please try again."