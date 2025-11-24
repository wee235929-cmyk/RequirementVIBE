"""
Session Management Service

This module handles creation, retrieval, and management of conversation sessions.
Each session maintains its own chat history, model selection, and metadata.
"""

import streamlit as st
import uuid
from datetime import datetime
from core.models.memory import ShortTermMemory


def create_new_session() -> str:
    """
    Create a new conversation session with unique ID.
    
    This function:
    1. Saves the current session's messages and model to persistent storage
    2. Generates a new unique session ID using UUID
    3. Creates a new session entry with empty messages
    4. Resets the memory for the new session
    5. Clears any generated SRS documents
    
    The new session uses the currently selected model, which can be changed
    before the first message is sent. After the first message, the model is locked.
    
    Returns:
        str: The newly created session ID (UUID string)
    
    Side Effects:
        - Updates st.session_state.sessions with new session
        - Sets st.session_state.current_session_id to new session
        - Resets st.session_state.memory to empty ShortTermMemory
        - Clears SRS generation state
    """
    # Generate unique session identifier using UUID
    session_id = str(uuid.uuid4())
    
    # Save current session's data before creating new one
    # This ensures no data is lost when switching between sessions
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        # Retrieve previous session and save current memory state
        prev_session = st.session_state.sessions[st.session_state.current_session_id]
        prev_session["messages"] = st.session_state.memory.get_messages()  # Save chat history
        prev_session["model"] = st.session_state.selected_model  # Save model used in this session
        st.session_state.sessions[st.session_state.current_session_id] = prev_session
    
    # Import greeting utility to assign a greeting to the new session
    from utils.greetings import get_random_greeting
    
    # Create new session entry with initial state
    st.session_state.sessions[session_id] = {
        "id": session_id,                                    # Unique identifier
        "messages": [],                                      # Empty chat history
        "title": f"New Chat {st.session_state.session_counter + 1}",  # Default title (updated from first message)
        "created_at": datetime.now().isoformat(),            # Timestamp for sorting (ISO format for JSON)
        "model": st.session_state.selected_model,           # Model selected for this session
        "greeting": get_random_greeting()                    # Random greeting for this session
    }
    st.session_state.session_counter += 1
    st.session_state.current_session_id = session_id
    
    # Reset memory for new session (fresh start)
    st.session_state.memory = ShortTermMemory()
    
    # Clear generated SRS when creating new session (SRS is session-specific)
    st.session_state.generated_srs = None
    st.session_state.srs_generation_error = None
    
    # Save conversations to disk if persistence is enabled
    if st.session_state.conversation_persistence_enabled and st.session_state.conversation_storage:
        st.session_state.conversation_storage.save_sessions(st.session_state.sessions)
    
    return session_id


def get_current_session() -> dict:
    """
    Get the currently active session, creating one if none exists.
    
    This function ensures there's always an active session. If no session exists
    (e.g., on first app load), it automatically creates a new one.
    
    Returns:
        dict: Session dictionary containing:
            - id: Session UUID
            - messages: List of chat messages
            - title: Session title (default or from first message)
            - created_at: Creation timestamp
            - model: Model used in this session
    """
    # Auto-create session if none exists (handles first-time app load)
    if st.session_state.current_session_id is None:
        create_new_session()
    
    # Retrieve and return the current session data
    session = st.session_state.sessions[st.session_state.current_session_id]
    return session


def update_session_title(session_id: str, first_message: str) -> None:
    """
    Update session title from the first user message.
    
    When a user sends their first message, this function extracts the first
    50 characters to use as a more descriptive session title, replacing
    the default "New Chat N" title.
    
    Args:
        session_id: UUID string identifying the session to update
        first_message: The first user message content (used to generate title)
    
    Side Effects:
        - Updates the session title in st.session_state.sessions if:
          - Session exists
          - Current title is still the default "New Chat" format
    """
    if session_id in st.session_state.sessions:
        # Only update if title is still the default (hasn't been manually changed)
        if st.session_state.sessions[session_id]["title"].startswith("New Chat"):
            # Extract first 50 characters as title
            title = first_message[:50]
            if len(first_message) > 50:
                title += "..."  # Add ellipsis if message was truncated
            st.session_state.sessions[session_id]["title"] = title
            # Save conversations to disk if persistence is enabled
            if st.session_state.conversation_persistence_enabled and st.session_state.conversation_storage:
                st.session_state.conversation_storage.save_sessions(st.session_state.sessions)

