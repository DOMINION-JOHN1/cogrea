from typing import Dict, List, TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from config import REDIS_HOST, REDIS_PASSWORD,llm, tavily_tool , REDIS_URL
from .custom_checkpointer import CustomRedisCheckpointer

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
    builder = StateGraph(AgentState)
    
    # Add the model node (unchanged)
    builder.add_node("call_model", call_model)
    
    # Rename this node to "tools" to match tools_condition's default branch
    builder.add_node("tools", ToolNode([tavily_tool]))
    
    # Conditional edges from call_model (unchanged; now points to "tools" or "__end__")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    
    # After tools are invoked, route back to call_model (fix the previous typo; no "chatbot" node exists)
    builder.add_edge("tools", "call_model")
    
    # Start at call_model (unchanged)
    builder.add_edge(START, "call_model")
    
    # Use the custom checkpointer (unchanged)
    checkpointer = CustomRedisCheckpointer()
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
    new_message = HumanMessage(content=user_message)
    
    # Use stream instead of invoke to better handle state updates
    try:
        # Stream the workflow to get the response
        events = list(workflow.stream(
            {"messages": [new_message]},
            config,
            stream_mode="values"
        ))
        
        # Get the final state
        if events:
            final_state = events[-1]
            if final_state and "messages" in final_state:
                messages = final_state["messages"]
                if messages and hasattr(messages[-1], 'content'):
                    return messages[-1].content
    except Exception as e:
        print(f"Error in workflow stream: {e}")
        
        # Fallback: try with invoke
        try:
            result = workflow.invoke({"messages": [new_message]}, config)
            if result["messages"] and hasattr(result["messages"][-1], 'content'):
                return result["messages"][-1].content
        except Exception as e2:
            print(f"Error in workflow invoke: {e2}")
    
    return "I'm sorry, I couldn't generate a response. Please try again."