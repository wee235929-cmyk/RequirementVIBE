"""
ReqVibe - AI Requirements Analyst Application

This Streamlit application provides an AI-powered requirements engineering assistant that:
- Analyzes and refines software requirements using Volere template structure
- Supports multiple LLM models (DeepSeek, GPT, Claude, Grok, Gemini) via centralized API
- Manages conversation sessions with persistent chat history
- Generates IEEE 830 SRS documents from conversations

Main Components:
1. Centralized LLM API Client - Wraps the centralized API to mimic OpenAI interface
2. Session Management - Handles multiple conversation sessions with model persistence
3. Memory Management - Manages chat history, token counting, and context window
4. UI Components - Sidebar for session/model management, main chat area
5. SRS Generation - Converts conversation history to IEEE 830 format documents
"""

# Add project root to Python path for Streamlit Cloud compatibility
# This MUST be done before any other imports
import sys
import os

def _find_project_root():
    """Find the project root by looking for app.py or requirements.txt."""
    # Start from this file's directory (app.py should be in project root)
    current = os.path.dirname(os.path.abspath(__file__))
    
    # Check if current directory is project root
    has_domain = os.path.exists(os.path.join(current, 'domain'))
    has_app = os.path.exists(os.path.join(current, 'app.py'))
    has_requirements = os.path.exists(os.path.join(current, 'requirements.txt'))
    
    if has_domain and (has_app or has_requirements):
        # Verify domain/conversations exists
        if os.path.exists(os.path.join(current, 'domain', 'conversations')):
            return current
    
    # Go up the directory tree looking for project root markers
    for _ in range(6):  # Max 6 levels up (to handle /mount/src/requirementvibe structure)
        current = os.path.dirname(current)
        has_domain = os.path.exists(os.path.join(current, 'domain'))
        has_app = os.path.exists(os.path.join(current, 'app.py'))
        has_requirements = os.path.exists(os.path.join(current, 'requirements.txt'))
        
        if has_domain and (has_app or has_requirements):
            # Verify domain/conversations exists
            if os.path.exists(os.path.join(current, 'domain', 'conversations')):
                return current
    
    # Fallback: use directory containing app.py
    return os.path.dirname(os.path.abspath(__file__))

# Find and add project root to sys.path
_project_root = _find_project_root()

# Add project root to sys.path with verification
if _project_root:
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    
    # Verify the path works by checking if domain package is accessible
    domain_path = os.path.join(_project_root, 'domain')
    if not os.path.exists(domain_path):
        # Try parent directory (for cases where repo is in a subdirectory)
        _parent = os.path.dirname(_project_root)
        if _parent and os.path.exists(os.path.join(_parent, 'domain')):
            if _parent not in sys.path:
                sys.path.insert(0, _parent)

# Load environment variables from .env file
# This must be done before any other imports that might use environment variables
from dotenv import load_dotenv

# Load .env file from project root directory (same directory as app.py)
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    # Only load once to avoid redundant parsing when Streamlit reruns scripts
    load_dotenv(dotenv_path=env_path, override=False)

import streamlit as st
import copy
import time

# Presentation Layer (UI)
from presentation.styles import apply_styles
from presentation.components.sidebar import render_sidebar

# Domain Services
from domain.sessions.service import create_new_session, get_current_session, update_session_title
from domain.requirements.service import extract_requirements_from_response, merge_requirement_with_pending
from domain.prompts.service import decide_and_build_prompt
from domain.conversations.service import ConversationStorage

# Infrastructure
from infrastructure.llm.client import get_centralized_client

# Core Models
from core.models.memory import ShortTermMemory

# Utils
from utils.state_manager import initialize_session_state
from utils.greetings import get_greeting_for_session
from monitoring.langsmith import traceable


@traceable(name="format_prompt", run_type="chain")
def format_prompt(user_message, history, requirements):
    """Build system prompt and extract new requirement data."""
    return decide_and_build_prompt(
        user_message=user_message,
        history=history,
        requirements=requirements
    )


@traceable(name="invoke_llm", run_type="llm")
def invoke_llm_response(
    client,
    model_name,
    messages,
    temperature=0.7,
    max_tokens=2000,
    stream=False,
    **kwargs,
):
    """Send the prompt to the centralized LLM API."""
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )


@traceable(name="parse_model_response")
def parse_model_response(response):
    """Extract assistant text from an LLM response object."""
    if not response:
        return ""
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
# Sets up the Streamlit page with title, icon, layout, and initial sidebar state
# The sidebar starts expanded to show session management and model selection
# Load icon for page config
from PIL import Image

# Get the path to the icon file
icon_path = os.path.join(os.path.dirname(__file__), "RequirementVIBEICON.png")
page_icon = None
if os.path.exists(icon_path):
    try:
        page_icon = Image.open(icon_path)
    except Exception as e:
        print(f"Could not load icon: {e}")
        page_icon = "📋"  # Fallback to emoji

st.set_page_config(
    page_title="Requirement Auto Analysis:UESTC-MBSE Requirement Assistant",
    page_icon=page_icon if page_icon else "📋",
    layout="wide",
    initial_sidebar_state="expanded"  # Start with sidebar expanded, but user can collapse it
)

# ----------------------------------------------------------------------
# UI Styling - ChatGPT-like Dark Theme
# ----------------------------------------------------------------------
# Apply custom CSS styling for dark theme
apply_styles()

# ----------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------
# Initialize all session state variables with default values
initialize_session_state()

# ----------------------------------------------------------------------
# Sidebar UI - Session and Model Management
# ----------------------------------------------------------------------
# Render sidebar with all UI components
# The sidebar component handles:
# 1. Model selection (with lock mechanism after session starts)
# 2. Session management (create new, switch between sessions)
# 3. SRS export functionality
# 4. File upload and document processing
# 5. Conversation persistence settings

render_sidebar()

# Ensure conversation storage is initialized for authenticated user
if st.session_state.authenticated and st.session_state.current_user and st.session_state.conversation_storage is None:
    st.session_state.conversation_storage = ConversationStorage(st.session_state.current_user["username"])
    # Load previous conversations if they exist (always load on initialization)
    loaded_sessions = st.session_state.conversation_storage.load_sessions()
    if loaded_sessions:
        # Merge with existing sessions (don't overwrite current sessions)
        for session_id, session_data in loaded_sessions.items():
            if session_id not in st.session_state.sessions:
                st.session_state.sessions[session_id] = session_data
        # Update session counter
        if st.session_state.sessions:
            st.session_state.session_counter = len(st.session_state.sessions)
        # Restore current session if available and no current session is set
        if st.session_state.sessions and not st.session_state.current_session_id:
            # Get the most recent session
            sorted_sessions = sorted(
                st.session_state.sessions.values(),
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
            if sorted_sessions:
                most_recent_session = sorted_sessions[0]
                st.session_state.current_session_id = most_recent_session["id"]
                # Load messages into memory
                if most_recent_session.get("messages"):
                    st.session_state.memory.load_messages(most_recent_session["messages"], reset=True)
                # Restore model
                if most_recent_session.get("model"):
                    st.session_state.selected_model = most_recent_session["model"]

# ----------------------------------------------------------------------
# Main Chat Interface
# ----------------------------------------------------------------------
# This section handles the main chat area where users interact with the AI
# It includes:
# 1. Chat input field
# 2. Message display
# 3. AI response generation
# 4. Session synchronization

# Chat input field - placed at bottom for better UX (Streamlit handles positioning)
user_input = st.chat_input("Ask for requirement analysis...")

# Get the current active session
current_session = get_current_session()

# Prevent unauthenticated users from sending chat messages
if user_input and not st.session_state.authenticated:
    st.session_state.sidebar_login_prompt = True
    st.warning("Please log in via the sidebar before starting a conversation.")
    user_input = None

# Synchronize memory with session storage on page load
if current_session["messages"] and st.session_state.memory.get_history_length() == 0:
    st.session_state.memory.load_messages(current_session["messages"], reset=True)

# Handle pending file upload message (from file upload component)
if st.session_state.get("pending_file_upload_message") and not user_input:
    # Use the pending message as user input
    user_input = st.session_state.pending_file_upload_message
    st.session_state.pending_file_upload_message = None  # Clear after use

# Handle pending voice transcription message (from voice input component)
if st.session_state.get("pending_voice_message") and not user_input:
    user_input = st.session_state.pending_voice_message
    st.session_state.pending_voice_message = None

# Handle user input when a message is submitted
if user_input:
    # Add user message to memory immediately (for display and context)
    st.session_state.memory.add_message("user", user_input)
    
    # Update session title from first user message
    if st.session_state.memory.get_history_length() == 1:
        update_session_title(st.session_state.current_session_id, user_input)

# Retrieve messages from memory for display
messages = st.session_state.memory.get_messages()

# Synchronize session storage with memory
if current_session["messages"] != messages:
    current_session["messages"] = messages
    current_session["model"] = st.session_state.selected_model
    st.session_state.sessions[st.session_state.current_session_id] = current_session
    
    # Save conversations to disk if persistence is enabled
    if st.session_state.conversation_persistence_enabled and st.session_state.conversation_storage and messages:
        st.session_state.conversation_storage.save_sessions(st.session_state.sessions)

# Display welcome message if conversation is empty
if len(messages) == 0:
    # Get the greeting for this session (assigns one if it doesn't exist)
    greeting = get_greeting_for_session(
        st.session_state.current_session_id,
        st.session_state.sessions
    )
    st.markdown(f"""
    <div style='text-align: center; padding: 4rem 1rem 2rem 1rem; max-width: 700px; margin: 0 auto;'>
        <h1 style='color: #ececf1; font-size: 2.75rem; font-weight: 600; margin-bottom: 1.5rem; line-height: 1.2;'>{greeting}</h1>
        <p style='color: #8e8ea0; font-size: 1.1rem; line-height: 1.6; margin: 0;'>Ask MBSE ReqViber to help you analyze and refine your software requirements using IEEE 830 SRS or Volere template structure.</p>
        <p style='color: #8e8ea0; font-size: 1.1rem; line-height: 1.6; margin: 0;'>If you have any questions about using the website or about the project requirements, please feel free to contact me at wee235929@gmail.com.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display all chat messages including the new user message
    for message in messages:
        with st.chat_message(message["role"]):
            # Use Mermaid renderer for assistant messages to detect and render diagrams
            if message["role"] == "assistant":
                from utils.renderers.mermaid import render_message_with_mermaid
                render_message_with_mermaid(message["content"])
            else:
                st.markdown(message["content"])

# ----------------------------------------------------------------------
# AI Response Generation
# ----------------------------------------------------------------------
# When a user sends a message, this section:
# 1. Prepares the conversation context (with token management)
# 2. Calls the LLM API with the selected model
# 3. Displays the response
# 4. Saves the response to memory
# 5. Extracts and saves requirements from the response

# Process AI response if user just sent a message
if user_input:
    try:
        # Get API client (centralized LLM API)
        client = get_centralized_client()
        
        # Check if GraphRAG should be used for this query
        use_graphrag = False
        graphrag_answer = None
        
        if st.session_state.get("graphrag_index_built") and st.session_state.get("graphrag_index"):
            from infrastructure.graphrag.service import (
                is_document_related_query,
                GraphRAGIndex,
                answer_question_with_graphrag
            )
            
            # Check if query is document-related
            if is_document_related_query(user_input):
                try:
                    # Reconstruct GraphRAG index from serialized data
                    index_data = st.session_state.graphrag_index
                    graphrag_index = GraphRAGIndex.from_dict(index_data)
                    
                    # Rebuild graph and embeddings (they weren't serialized)
                    from infrastructure.graphrag.service import build_graphrag_index
                    if st.session_state.get("document_processing_results"):
                        graphrag_index = build_graphrag_index(
                            st.session_state.document_processing_results
                        )
                    
                    # Try to answer using GraphRAG
                    with st.chat_message("assistant"):
                        with st.spinner("Searching documents..."):
                            graphrag_answer = answer_question_with_graphrag(
                                query=user_input,
                                index=graphrag_index,
                                llm_client=client,
                                model=st.session_state.selected_model
                            )
                            # Display the GraphRAG answer with Mermaid support
                            from utils.renderers.mermaid import render_message_with_mermaid
                            render_message_with_mermaid(graphrag_answer)
                            use_graphrag = True
                except Exception as e:
                    # If GraphRAG fails, fall back to normal chat
                    print(f"GraphRAG error: {str(e)}. Falling back to normal chat.")
                    use_graphrag = False
        
        # If not using GraphRAG, use normal chat flow
        if not use_graphrag:
            # Get current requirements from memory (short-term)
            current_requirements = st.session_state.memory.get_requirements()
            
            # Get conversation history for prompt building
            history = st.session_state.memory.get_messages(include_system=False)
            
            # Use decide_and_build_prompt to build proper prompt and detect requirements
            api_messages, conflict_message, new_requirement_data = format_prompt(
                user_message=user_input,
                history=history,
                requirements=current_requirements
            )
            
            # Store pending requirement from user input (will be saved after assistant response)
            if new_requirement_data:
                st.session_state.pending_requirement = copy.deepcopy(new_requirement_data)
            else:
                st.session_state.pending_requirement = None
            
            # Get conversation context from memory with automatic token management
            context_messages = st.session_state.memory.get_context_for_api(
                max_tokens=3500,
                client=client,
                model=st.session_state.selected_model
            )
            
            # Replace history in api_messages with context_messages (which may be truncated)
            if api_messages and api_messages[0].get("role") == "system":
                system_prompt = api_messages[0]["content"]
                api_messages = [{"role": "system", "content": system_prompt}]
                api_messages.extend(context_messages)
                # Add current user message if not already in context_messages
                if not any(msg.get("content") == user_input for msg in context_messages):
                    api_messages.append({"role": "user", "content": user_input})
            else:
                # Fallback: use simple system prompt
                system_prompt = "You are ReqViber, a professional requirements engineer. Use Volere template structure."
                api_messages = [{"role": "system", "content": system_prompt}]
                api_messages.extend(context_messages)
            
            # Display streaming response from the LLM API
            with st.chat_message("assistant"):
                message_container = st.container()
                stream_placeholder = message_container.empty()
                streamed_text = ""
                last_update_time = 0.0
                try:
                    stream = invoke_llm_response(
                        client=client,
                        model_name=st.session_state.selected_model,
                        messages=api_messages,
                        temperature=0.7,
                        max_tokens=2000,
                        stream=True,
                    )
                    
                    for chunk in stream:
                        if not chunk:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            # Ensure content is properly decoded as UTF-8 string
                            if isinstance(content, bytes):
                                content = content.decode('utf-8', errors='replace')
                            streamed_text += content
                            current_time = time.time()
                            if (
                                current_time - last_update_time >= 0.05
                                or content.endswith((".", "!", "?", "\n"))
                            ):
                                stream_placeholder.markdown(streamed_text + "▌")
                                last_update_time = current_time
                    
                    stream_placeholder.empty()
                    ai_response = streamed_text.strip() or "I couldn't generate a response. Please try again."
                    from utils.renderers.mermaid import render_message_with_mermaid
                    with message_container:
                        render_message_with_mermaid(ai_response)
                except Exception as stream_error:
                    stream_placeholder.empty()
                    print(f"Streaming error: {stream_error}")
                    st.warning(
                        "Streaming isn't available right now. Showing the full response instead.",
                        icon="⚠️",
                    )
                    with st.spinner("Analyzing requirement..."):
                        response = invoke_llm_response(
                            client=client,
                            model_name=st.session_state.selected_model,
                            messages=api_messages,
                            temperature=0.7,
                            max_tokens=2000,
                            stream=False,
                        )
                        ai_response = parse_model_response(response) or "I couldn't generate a response. Please try again."
                        from utils.renderers.mermaid import render_message_with_mermaid
                        with message_container:
                            render_message_with_mermaid(ai_response)
        else:
            # Using GraphRAG answer
            ai_response = graphrag_answer
        
        # Save AI response to memory for future context and display
        st.session_state.memory.add_message("assistant", ai_response)
        
        # Only extract requirements if not using GraphRAG (GraphRAG answers are document-focused)
        if not use_graphrag:
            # Extract and save requirements from AI response using requirement service
            pending_requirement = st.session_state.pending_requirement
            existing_requirements = st.session_state.memory.get_requirements()
            
            # Extract requirements from AI response
            extracted_requirements = extract_requirements_from_response(
                ai_response,
                existing_requirements=existing_requirements
            )
            
            # Save each extracted requirement
            for requirement_data in extracted_requirements:
                # Merge with pending requirement if it matches
                original_req_id = requirement_data.get("id", "").upper()
                if pending_requirement and pending_requirement.get("id", "").upper() == original_req_id:
                    requirement_data = merge_requirement_with_pending(
                        requirement_data,
                        pending_requirement,
                        original_req_id
                    )
                    st.session_state.pending_requirement = None
                
                # Save to memory
                try:
                    st.session_state.memory.add_requirement(requirement_data)
                    print(f"DEBUG: Successfully saved requirement {requirement_data.get('id')} to memory")
                except Exception as e:
                    print(f"ERROR: Failed to save requirement {requirement_data.get('id')}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        st.rerun()  # Refresh UI to show the new message
        
    except Exception as e:
        # Handle errors gracefully - display error message to user
        error_msg = f"Sorry, an error occurred: {str(e)}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        
        # Save error message to memory so it appears in conversation history
        st.session_state.memory.add_message("assistant", error_msg)
        st.rerun()

