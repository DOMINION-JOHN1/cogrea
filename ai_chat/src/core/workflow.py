from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode, tools_condition
from config import llm, tavily_tool
from .custom_checkpointer import CustomRedisCheckpointer
from prompts.prompts import system_prompt

# Define the state for our conversation
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda a, b: a + b]

# System prompt imported from prompts.py
SYSTEM_PROMPT = system_prompt

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
    
    print(f"DEBUG: Processing message for session {session_id}")
    print(f"DEBUG: User message: {user_message}")
    
    # Check current state before processing
    try:
        current_state = workflow.get_state(config)
        print(f"DEBUG: Current state exists: {current_state is not None}")
        if current_state and hasattr(current_state, 'values') and current_state.values:
            print(f"DEBUG: Current state values: {current_state.values}")
            if "messages" in current_state.values:
                existing_messages = current_state.values["messages"]
                print(f"DEBUG: Found {len(existing_messages)} existing messages")
                for i, msg in enumerate(existing_messages):
                    print(f"DEBUG: Message {i}: {type(msg).__name__} - {msg.content[:100]}...")
            else:
                print("DEBUG: No 'messages' key in current state")
        else:
            print("DEBUG: No current state or state values")
    except Exception as e:
        print(f"DEBUG: Error getting current state: {e}")
    
    # Create a human message from the user input
    new_message = HumanMessage(content=user_message)
    
    # Use stream to process the message
    try:
        print("DEBUG: Starting workflow stream...")
        events = list(workflow.stream(
            {"messages": [new_message]},
            config,
            stream_mode="values"
        ))
        
        print(f"DEBUG: Got {len(events)} events from stream")
        
        # Get the final state
        if events:
            final_state = events[-1]
            print(f"DEBUG: Final state: {final_state}")
            if final_state and "messages" in final_state:
                messages = final_state["messages"]
                print(f"DEBUG: Final state has {len(messages)} messages")
                if messages and hasattr(messages[-1], 'content'):
                    response_content = messages[-1].content
                    print(f"DEBUG: Response: {response_content}")
                    
                    # Check state after processing
                    try:
                        post_state = workflow.get_state(config)
                        if post_state and hasattr(post_state, 'values') and post_state.values:
                            print(f"DEBUG: Post-processing state has {len(post_state.values.get('messages', []))} messages")
                    except Exception as e:
                        print(f"DEBUG: Error checking post-processing state: {e}")
                    
                    return response_content
    except Exception as e:
        print(f"DEBUG: Error in workflow stream: {e}")
        import traceback
        traceback.print_exc()
    
    return "I'm sorry, I couldn't generate a response. Please try again."