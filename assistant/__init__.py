"""
AeroScript JARVIS Voice Assistant
==================================
Provides voice-controlled interaction for AeroScript via:
  - Speech-to-Text  (SpeechRecognition + Google STT)
  - Text-to-Speech  (pyttsx3)
  - Intent-based Command Processing
  - Push-to-Talk (V key) and optional Wake-Word ("Hey Jarvis")

All voice processing runs in background threads and communicates
with the OpenCV main loop exclusively via a thread-safe queue.
"""

from .jarvis import JarvisAssistant
from .command_processor import VoiceCommand, Intent

__all__ = ["JarvisAssistant", "VoiceCommand", "Intent"]
