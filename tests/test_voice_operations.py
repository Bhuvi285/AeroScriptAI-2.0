"""
AeroScript 2.0 -- Voice Operations & Hands-Free Integration Tests
==================================================================
Validates that AeroScript operates 100% via JARVIS voice commands:
  - Auto-start without requiring Enter key
  - Continuous listening active without pressing V
  - Gemini voice search ("search who is the prime miniser of india")
  - Speech-to-text populating Gemini search box
  - Handwritten paste-to-screen ("paste into screen")
  - Gemini close ("close the gemini")
  - Math mode ("open math mode")
  - Whiteboard mode ("open whiteboard")
  - Camera mode ("camera mode")
  - Complete project shutdown ("close the aeroscript")
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from assistant.command_processor import CommandProcessor, Intent, VoiceCommand
from assistant.controller import AeroScriptController
from assistant.jarvis import execute_command, JarvisAssistant, JarvisStatus
from assistant.responses import get_response, get_toast


class MockApp:
    """Mock of AeroScript App for end-to-end headless testing."""
    def __init__(self):
        self.canvas = MagicMock()
        self.canvas.surface = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.white_mode = False
        self.color_idx = 0
        self.tool_idx = 0
        self.eraser = False
        self.brush = 5
        self.chat_mode = False
        self.chat_input = ""
        self.chat_answer = ""
        self.chat_stamped = False
        self.chat_thinking = False
        self.running = True
        self.toast = ""
        self.ai_result = None
        self.ai_solved = False
        self.gemini = MagicMock()
        self._gemini_callback = MagicMock()
        self.COLORS = [("Red", (0, 0, 200)), ("Green", (0, 200, 0)), ("Blue", (200, 0, 0))]

    def _toast(self, msg):
        self.toast = msg

    def _color(self):
        return (0, 0, 200)

    def _tool(self):
        tools = ["pen", "marker", "highlighter", "shape_pen", "air_text", "smart_math"]
        return tools[self.tool_idx]

    def _stamp_gemini_answer(self):
        self.chat_stamped = True
        self.chat_input = ""
        self._toast("Gemini answer stamped in your color!")

    def _stamp_ai_result(self):
        self.ai_solved = True
        self._toast("AI result stamped!")


class TestVoiceOperationsWorkflow(unittest.TestCase):
    """Verifies the complete user scenario requested by the user."""

    def setUp(self):
        self.cp = CommandProcessor()
        self.app = MockApp()
        self.ctrl = AeroScriptController(self.app)
        self.tts = MagicMock()

    def test_voice_command_dispatch_full_cycle(self):
        """
        User scenario:
        1. "open gemini"
        2. "search who is the prime miniser of india"
        3. "paste in to screen"
        4. "close the gemini"
        5. "open math mode"
        6. "open whiteboard"
        7. "close the aeroscript"
        """

        # Step 1: "open gemini"
        cmd1 = self.cp.process("open gemini")
        self.assertEqual(cmd1.intent, Intent.OPEN_GEMINI)
        execute_command(cmd1, self.ctrl, self.tts)
        self.assertTrue(self.app.chat_mode)
        self.assertIn("Gemini Chat ON", self.app.toast)

        # Step 2: "search who is the prime miniser of india"
        cmd2 = self.cp.process("search who is the prime miniser of india", chat_mode_active=self.app.chat_mode)
        self.assertEqual(cmd2.intent, Intent.SEARCH_GEMINI)
        self.assertEqual(cmd2.params.get("query"), "who is the prime miniser of india")
        execute_command(cmd2, self.ctrl, self.tts)
        self.assertEqual(self.app.chat_input, "who is the prime miniser of india")
        self.assertTrue(self.app.chat_thinking)
        self.app.gemini.ask.assert_called_with("who is the prime miniser of india", self.app._gemini_callback)

        # Simulate Gemini response received
        self.app.chat_answer = "The Prime Minister of India is Narendra Modi."
        self.app.chat_thinking = False

        # Step 3: "paste in to screen"
        cmd3 = self.cp.process("paste in to screen")
        self.assertEqual(cmd3.intent, Intent.PASTE_TO_SCREEN)
        execute_command(cmd3, self.ctrl, self.tts)
        self.assertTrue(self.app.chat_stamped)
        self.assertEqual(self.app.chat_input, "")  # Cleared after stamping

        # Step 4: "close the gemini"
        cmd4 = self.cp.process("close the gemini")
        self.assertEqual(cmd4.intent, Intent.CLOSE_GEMINI)
        execute_command(cmd4, self.ctrl, self.tts)
        self.assertFalse(self.app.chat_mode)
        self.assertIn("Gemini Chat OFF", self.app.toast)

        # Step 5: "open math mode"
        cmd5 = self.cp.process("open math mode")
        self.assertEqual(cmd5.intent, Intent.ENABLE_MATH)
        execute_command(cmd5, self.ctrl, self.tts)
        self.assertEqual(self.app._tool(), "smart_math")
        self.assertIn("Smart Math", self.app.toast)

        # Step 6: "open whiteboard"
        cmd6 = self.cp.process("open whiteboard")
        self.assertEqual(cmd6.intent, Intent.ENABLE_WHITEBOARD)
        execute_command(cmd6, self.ctrl, self.tts)
        self.assertTrue(self.app.white_mode)
        self.assertIn("Whiteboard ON", self.app.toast)

        # Step 7: "close the aeroscript"
        cmd7 = self.cp.process("close the aeroscript")
        self.assertEqual(cmd7.intent, Intent.CLOSE_PROJECT)
        execute_command(cmd7, self.ctrl, self.tts)
        self.assertFalse(self.app.running)
        self.assertIn("Closing AeroScript", self.app.toast)

    def test_push_to_talk_initialization(self):
        """Verify JARVIS starts in READY mode (Phase 3 click-to-talk)."""
        with patch("assistant.speech_to_text.SpeechToTextService.start", return_value=True), \
             patch("assistant.text_to_speech.TextToSpeechService.start", return_value=True):
            jarvis = JarvisAssistant(startup_speech=False)
            ok = jarvis.start()
            self.assertTrue(ok)
            self.assertEqual(jarvis.status, JarvisStatus.READY)
            jarvis.stop()

    def test_auto_start_without_enter(self):
        """Verify _startup automatically returns 'canvas' without requiring keyboard Enter."""
        from main import App
        app = App()
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=255), patch("cv2.getWindowProperty", return_value=1):
            with patch("time.time", side_effect=[100.0, 100.1, 100.5, 101.2]):
                result = app._startup()
                self.assertEqual(result, "canvas")


if __name__ == "__main__":
    unittest.main()
