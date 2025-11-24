"""
Greeting Utilities for ReqVibe

This module provides functions to generate and manage welcome greetings
that appear when users start a new conversation or open an empty session.
"""

import random


# Collection of diverse, friendly greetings for the welcome screen
GREETINGS = [
    "What can I help with? Ready when you are!",
    "What are you working on?",
    "How can I assist you today?",
    "What would you like to explore?",
    "Ready to dive into your requirements?",
    "What's on your mind?",
    "How can I help you get started?",
    "What project are we tackling today?",
    "What requirements are you analyzing?",
    "Ready to refine your requirements?",
    "What can we build together?",
    "How can I support your work?",
    "What are you looking to accomplish?",
    "What's your next challenge?",
    "How can I help you succeed?",
]


def get_random_greeting() -> str:
    """
    Generate a random greeting from the available collection.
    
    Returns:
        str: A randomly selected greeting message
    """
    return random.choice(GREETINGS)


def get_greeting_for_session(session_id: str, sessions: dict) -> str:
    """
    Get the greeting for a specific session, assigning one if it doesn't exist.
    
    This function ensures each session has its own greeting that persists
    throughout the session. If a greeting hasn't been assigned yet, a new
    one is randomly selected and stored in the session data.
    
    Args:
        session_id: The UUID string identifying the session
        sessions: Dictionary mapping session_id -> session data
    
    Returns:
        str: The greeting message for this session
    """
    # Check if session exists and has a greeting
    if session_id in sessions:
        session = sessions[session_id]
        if "greeting" in session:
            return session["greeting"]
        
        # Assign a new greeting if one doesn't exist
        greeting = get_random_greeting()
        session["greeting"] = greeting
        return greeting
    
    # Fallback: return a random greeting if session doesn't exist
    return get_random_greeting()

