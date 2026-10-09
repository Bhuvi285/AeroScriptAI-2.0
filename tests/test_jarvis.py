"""
AeroScript JARVIS — Unit Tests
================================
Tests command parsing, color extraction, canvas commands, tool commands,
mode commands, unknown commands, and queue communication.

NO microphone required — speech recognition is fully mocked.

Run with:
    python3 -m pytest tests/test_jarvis.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import queue
import unittest
from unittest.mock import MagicMock, patch

from assistant.command_processor import CommandProcessor, VoiceCommand, Intent
from assistant.responses import get_response, get_toast


# ─── CommandProcessor Tests ───────────────────────────────────────────────────

class TestCommandProcessor(unittest.TestCase):
    """Tests for the intent-based command processor."""

    def setUp(self):
        self.cp = CommandProcessor()

    # ── Canvas commands ───────────────────────────────────────────────────────

    def test_clear_canvas_direct(self):
        cmd = self.cp.process("Jarvis, clear the canvas")
        self.assertEqual(cmd.intent, Intent.CLEAR_CANVAS)

    def test_clear_canvas_variations(self):
        phrases = [
            "clear canvas",
            "clear everything",
            "Jarvis clear everything",
            "please clear my canvas",
            "erase canvas",
            "wipe canvas",
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                cmd = self.cp.process(phrase)
                self.assertEqual(cmd.intent, Intent.CLEAR_CANVAS, f"Failed for: '{phrase}'")

    def test_undo(self):
        for phrase in ["undo", "Jarvis undo", "go back", "step back"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.UNDO, f"Failed for: '{phrase}'")

    def test_redo(self):
        for phrase in ["redo", "Jarvis redo", "redo that"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.REDO, f"Failed for: '{phrase}'")

    def test_save_canvas(self):
        for phrase in ["save canvas", "save the drawing", "save image", "export canvas"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.SAVE_CANVAS, f"Failed for: '{phrase}'")

    def test_screenshot(self):
        for phrase in ["take a screenshot", "screenshot", "capture screen"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.SCREENSHOT, f"Failed for: '{phrase}'")

    # ── Color commands ────────────────────────────────────────────────────────

    def test_change_color_blue(self):
        for phrase in ["change color to blue", "make it blue", "Jarvis use blue", "select blue"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.CHANGE_COLOR, f"Failed for: '{phrase}'")
            self.assertEqual(cmd.params.get("color_name"), "Blue", f"Wrong color for: '{phrase}'")
            self.assertEqual(cmd.params.get("color_idx"), 2, f"Wrong idx for: '{phrase}'")

    def test_change_color_red(self):
        cmd = self.cp.process("change color to red")
        self.assertEqual(cmd.intent, Intent.CHANGE_COLOR)
        self.assertEqual(cmd.params.get("color_idx"), 0)
        self.assertEqual(cmd.params.get("color_name"), "Red")

    def test_change_color_green(self):
        cmd = self.cp.process("Jarvis, change to green")
        self.assertEqual(cmd.intent, Intent.CHANGE_COLOR)
        self.assertEqual(cmd.params.get("color_idx"), 1)

    def test_change_color_yellow(self):
        cmd = self.cp.process("use yellow")
        self.assertEqual(cmd.intent, Intent.CHANGE_COLOR)
        self.assertEqual(cmd.params.get("color_idx"), 3)

    def test_change_color_orange(self):
        cmd = self.cp.process("switch to orange")
        self.assertEqual(cmd.intent, Intent.CHANGE_COLOR)
        self.assertEqual(cmd.params.get("color_idx"), 4)

    def test_change_color_white(self):
        cmd = self.cp.process("make it white")
        self.assertEqual(cmd.intent, Intent.CHANGE_COLOR)
        self.assertEqual(cmd.params.get("color_idx"), 5)

    def test_change_color_black(self):
        cmd = self.cp.process("use black")
        self.assertEqual(cmd.intent, Intent.CHANGE_COLOR)
        self.assertEqual(cmd.params.get("color_idx"), 6)

    # ── Brush commands ────────────────────────────────────────────────────────

    def test_increase_brush(self):
        for phrase in ["increase brush", "bigger brush", "brush up", "make brush bigger", "thicker"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.INCREASE_BRUSH, f"Failed for: '{phrase}'")

    def test_decrease_brush(self):
        for phrase in ["decrease brush", "smaller brush", "brush down", "thinner"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.DECREASE_BRUSH, f"Failed for: '{phrase}'")

    # ── Tool commands ─────────────────────────────────────────────────────────

    def test_select_eraser(self):
        for phrase in ["select eraser", "eraser", "use eraser", "erase mode"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.SELECT_ERASER, f"Failed for: '{phrase}'")

    def test_select_pen(self):
        for phrase in ["select pen", "use pen", "pen mode", "drawing mode"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.SELECT_PEN, f"Failed for: '{phrase}'")

    def test_select_highlighter(self):
        cmd = self.cp.process("select highlighter")
        self.assertEqual(cmd.intent, Intent.SELECT_HIGHLIGHTER)

    def test_select_marker(self):
        cmd = self.cp.process("use marker")
        self.assertEqual(cmd.intent, Intent.SELECT_MARKER)

    # ── Mode commands ─────────────────────────────────────────────────────────

    def test_enable_whiteboard(self):
        for phrase in ["enable whiteboard mode", "whiteboard", "white background", "white mode"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.ENABLE_WHITEBOARD, f"Failed for: '{phrase}'")

    def test_enable_camera_mode(self):
        for phrase in ["enable camera mode", "camera mode", "camera view"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.ENABLE_CAMERA_MODE, f"Failed for: '{phrase}'")

    def test_pause(self):
        for phrase in ["pause", "freeze", "hold on"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.PAUSE, f"Failed for: '{phrase}'")

    def test_resume(self):
        for phrase in ["resume", "continue", "unpause"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.RESUME, f"Failed for: '{phrase}'")

    def test_show_status(self):
        cmd = self.cp.process("show status")
        self.assertEqual(cmd.intent, Intent.SHOW_STATUS)

    def test_show_fps(self):
        for phrase in ["show fps", "what is the fps", "frames per second", "current fps"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.SHOW_FPS, f"Failed for: '{phrase}'")

    def test_direct_colors(self):
        for color, idx in [("blue", 2), ("red", 0), ("green", 1), ("yellow", 3), ("orange", 4), ("white", 5), ("black", 6)]:
            cmd = self.cp.process(color)
            self.assertEqual(cmd.intent, Intent.CHANGE_COLOR, f"Failed for '{color}'")
            self.assertEqual(cmd.params["color_idx"], idx)

    def test_smart_math_and_air_text_commands(self):
        self.assertEqual(self.cp.process("smart math").intent, Intent.ENABLE_MATH)
        self.assertEqual(self.cp.process("math mode").intent, Intent.ENABLE_MATH)
        self.assertEqual(self.cp.process("open math mode").intent, Intent.ENABLE_MATH)
        self.assertEqual(self.cp.process("open math").intent, Intent.ENABLE_MATH)
        self.assertEqual(self.cp.process("exit math").intent, Intent.EXIT_MATH)
        self.assertEqual(self.cp.process("close math mode").intent, Intent.EXIT_MATH)
        self.assertEqual(self.cp.process("exit math mode").intent, Intent.EXIT_MATH)
        self.assertEqual(self.cp.process("air text").intent, Intent.ENABLE_AIR_TEXT)
        self.assertEqual(self.cp.process("exit text").intent, Intent.EXIT_AIR_TEXT)
        self.assertEqual(self.cp.process("stamp").intent, Intent.STAMP_RESULT)
        self.assertEqual(self.cp.process("stamp result").intent, Intent.STAMP_RESULT)
        self.assertEqual(self.cp.process("print on screen").intent, Intent.STAMP_RESULT)
        self.assertEqual(self.cp.process("open gemini").intent, Intent.OPEN_GEMINI)
        self.assertEqual(self.cp.process("gemini").intent, Intent.OPEN_GEMINI)
        self.assertEqual(self.cp.process("open chat").intent, Intent.OPEN_GEMINI)
        self.assertEqual(self.cp.process("open gem").intent, Intent.OPEN_GEMINI)
        self.assertEqual(self.cp.process("open gmail").intent, Intent.OPEN_GEMINI)
        self.assertEqual(self.cp.process("open jamini").intent, Intent.OPEN_GEMINI)
        self.assertEqual(self.cp.process("close gemini").intent, Intent.CLOSE_GEMINI)
        self.assertEqual(self.cp.process("close the gemini").intent, Intent.CLOSE_GEMINI)
        self.assertEqual(self.cp.process("close gem").intent, Intent.CLOSE_GEMINI)
        self.assertEqual(self.cp.process("close gmail").intent, Intent.CLOSE_GEMINI)
        self.assertEqual(self.cp.process("exit chat").intent, Intent.CLOSE_GEMINI)

    def test_search_gemini_commands(self):
        cmd1 = self.cp.process("search who is the prime miniser of india")
        self.assertEqual(cmd1.intent, Intent.SEARCH_GEMINI)
        self.assertEqual(cmd1.params.get("query"), "who is the prime miniser of india")

        cmd2 = self.cp.process("search who is the prime minister of india")
        self.assertEqual(cmd2.intent, Intent.SEARCH_GEMINI)
        self.assertEqual(cmd2.params.get("query"), "who is the prime minister of india")

        cmd3 = self.cp.process("ask gemini what is photosynthesis")
        self.assertEqual(cmd3.intent, Intent.SEARCH_GEMINI)
        self.assertEqual(cmd3.params.get("query"), "what is photosynthesis")

        # Conversational fallback when chat_mode_active is True
        cmd4 = self.cp.process("who invented python", chat_mode_active=True)
        self.assertEqual(cmd4.intent, Intent.SEARCH_GEMINI)
        self.assertEqual(cmd4.params.get("query"), "who invented python")

    def test_paste_to_screen_commands(self):
        for phrase in [
            "paste into screen", "paste in to screen", "paste to screen",
            "paste on screen", "paste on canvas", "paste answer", "paste",
        ]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.PASTE_TO_SCREEN, f"Failed for: '{phrase}'")

    def test_close_project_commands(self):
        for phrase in [
            "close the aeroscript", "close aeroscript", "close project",
            "close the project", "exit aeroscript", "quit aeroscript",
        ]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.CLOSE_PROJECT, f"Failed for: '{phrase}'")

    def test_whiteboard_voice_variations(self):
        for phrase in ["white board", "whiteboard", "open whiteboard", "open white board", "whiteboard mode"]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.ENABLE_WHITEBOARD, f"Failed for: '{phrase}'")

    # ── Unknown command ───────────────────────────────────────────────────────

    def test_unknown_command(self):
        for phrase in ["xyzzy", "abracadabra", "make me a sandwich", ""]:
            cmd = self.cp.process(phrase)
            self.assertEqual(cmd.intent, Intent.UNKNOWN, f"Should be UNKNOWN for: '{phrase}'")

    def test_raw_text_preserved(self):
        raw = "Jarvis, change color to blue please"
        cmd = self.cp.process(raw)
        self.assertEqual(cmd.raw_text, raw)

    # ── Normalization ─────────────────────────────────────────────────────────

    def test_jarvis_prefix_stripped(self):
        """'Jarvis, ...' prefix should not prevent intent matching."""
        cmd = self.cp.process("Jarvis, please clear the canvas")
        self.assertEqual(cmd.intent, Intent.CLEAR_CANVAS)

    def test_case_insensitive(self):
        cmd = self.cp.process("CLEAR CANVAS")
        self.assertEqual(cmd.intent, Intent.CLEAR_CANVAS)


# ─── Responses Tests ──────────────────────────────────────────────────────────

class TestResponses(unittest.TestCase):

    def test_clear_response(self):
        resp = get_response(Intent.CLEAR_CANVAS)
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_color_response_substitution(self):
        resp = get_response(Intent.CHANGE_COLOR, color="Blue")
        self.assertIn("Blue", resp)

    def test_unknown_response(self):
        resp = get_response(Intent.UNKNOWN)
        self.assertIn("didn't understand", resp)

    def test_toast_substitution(self):
        toast = get_toast(Intent.CHANGE_COLOR, color="Red")
        self.assertIn("Red", toast)

    def test_all_intents_have_response(self):
        for intent in Intent:
            resp = get_response(intent)
            self.assertIsInstance(resp, str, f"No response for {intent}")


# ─── Queue Communication Tests ────────────────────────────────────────────────

class TestQueueCommunication(unittest.TestCase):
    """Verify thread-safe command queue mechanics without any real audio."""

    def test_drain_empty_queue(self):
        q = queue.Queue()
        cmds = []
        while not q.empty():
            cmds.append(q.get_nowait())
        self.assertEqual(cmds, [])

    def test_queue_put_get(self):
        q = queue.Queue()
        cmd = VoiceCommand(intent=Intent.CLEAR_CANVAS, raw_text="test")
        q.put(cmd)
        result = q.get_nowait()
        self.assertEqual(result.intent, Intent.CLEAR_CANVAS)

    def test_multiple_commands_in_order(self):
        q = queue.Queue()
        intents = [Intent.CLEAR_CANVAS, Intent.UNDO, Intent.SAVE_CANVAS]
        for i in intents:
            q.put(VoiceCommand(intent=i))
        drained = []
        while not q.empty():
            drained.append(q.get_nowait().intent)
        self.assertEqual(drained, intents)


# ─── AeroScriptController Tests (mocked App) ─────────────────────────────────

class TestAeroScriptController(unittest.TestCase):
    """Test controller methods against a mocked App."""

    def _make_mock_app(self):
        app = MagicMock()
        app.canvas = MagicMock()
        app.canvas.undo.return_value = True
        app.canvas.redo.return_value = True
        app.color_idx = 0
        app.eraser    = False
        app.white_mode = False
        app.brush     = 5
        app.tool_idx  = 0
        app.frame_ts  = [1, 2, 3]
        app._toast    = MagicMock()
        app._save     = MagicMock()
        app._tool     = MagicMock(return_value="pen")
        app.COLORS    = [("Red",), ("Green",), ("Blue",), ("Yellow",), ("Orange",), ("White",), ("Black",)]
        return app

    def setUp(self):
        from assistant.controller import AeroScriptController
        self.app = self._make_mock_app()
        self.ctrl = AeroScriptController(self.app)

    def test_clear_canvas(self):
        result = self.ctrl.clear_canvas()
        self.assertTrue(result)
        self.app.canvas.clear.assert_called_once()
        self.app._toast.assert_called_once()

    def test_undo(self):
        result = self.ctrl.undo()
        self.assertFalse(result)
        self.app._toast.assert_called_with("JARVIS: Undo is disabled")

    def test_redo(self):
        result = self.ctrl.redo()
        self.assertFalse(result)
        self.app._toast.assert_called_with("JARVIS: Redo is disabled")

    def test_save(self):
        result = self.ctrl.save_canvas()
        self.assertTrue(result)
        self.app._save.assert_called_once()

    def test_set_color_by_name(self):
        result = self.ctrl.set_color(color_name="blue")
        self.assertTrue(result)
        self.assertEqual(self.app.color_idx, 2)
        self.assertFalse(self.app.eraser)

    def test_set_color_by_idx(self):
        result = self.ctrl.set_color(color_idx=0)
        self.assertTrue(result)
        self.assertEqual(self.app.color_idx, 0)

    def test_set_color_invalid(self):
        result = self.ctrl.set_color(color_name="ultraviolet")
        self.assertFalse(result)

    def test_increase_brush(self):
        self.app.brush = 10
        result = self.ctrl.increase_brush()
        self.assertTrue(result)
        self.assertEqual(self.app.brush, 12)

    def test_decrease_brush(self):
        self.app.brush = 10
        result = self.ctrl.decrease_brush()
        self.assertTrue(result)
        self.assertEqual(self.app.brush, 8)

    def test_brush_max_clamp(self):
        self.app.brush = 39
        self.ctrl.increase_brush()
        self.assertEqual(self.app.brush, 40)

    def test_brush_min_clamp(self):
        self.app.brush = 3
        self.ctrl.decrease_brush()
        self.assertEqual(self.app.brush, 2)

    def test_select_eraser(self):
        result = self.ctrl.select_tool("eraser")
        self.assertTrue(result)
        self.assertTrue(self.app.eraser)

    def test_select_pen(self):
        self.app.eraser = True
        result = self.ctrl.select_tool("pen")
        self.assertTrue(result)
        self.assertFalse(self.app.eraser)
        self.assertEqual(self.app.tool_idx, 0)

    def test_enable_whiteboard(self):
        self.app.white_mode = False
        result = self.ctrl.enable_whiteboard()
        self.assertTrue(result)
        self.assertTrue(self.app.white_mode)

    def test_enable_camera_mode(self):
        self.app.white_mode = True
        result = self.ctrl.enable_camera_mode()
        self.assertTrue(result)
        self.assertFalse(self.app.white_mode)

    def test_search_gemini_controller(self):
        self.app.gemini = MagicMock()
        self.app.gemini.ask = MagicMock()
        result = self.ctrl.search_gemini("who is the prime minister of india")
        self.assertTrue(result)
        self.assertTrue(self.app.chat_mode)
        self.assertEqual(self.app.chat_input, "who is the prime minister of india")
        self.assertTrue(self.app.chat_thinking)
        self.app.gemini.ask.assert_called_once()

    def test_paste_to_screen_controller(self):
        self.app.chat_answer = "Narendra Modi is the Prime Minister of India."
        self.app._stamp_gemini_answer = MagicMock()
        result = self.ctrl.paste_to_screen()
        self.assertTrue(result)
        self.app._stamp_gemini_answer.assert_called_once()

    def test_close_project_controller(self):
        self.app.running = True
        result = self.ctrl.close_project()
        self.assertTrue(result)
        self.assertFalse(self.app.running)

    def test_controller_without_app(self):
        from assistant.controller import AeroScriptController
        ctrl = AeroScriptController(None)
        self.assertFalse(ctrl.clear_canvas())
        self.assertFalse(ctrl.undo())
        self.assertFalse(ctrl.set_color(color_name="blue"))
        self.assertFalse(ctrl.search_gemini("test"))
        self.assertFalse(ctrl.paste_to_screen())
        self.assertFalse(ctrl.close_project())


# ─── Speech Domain Correction Tests ───────────────────────────────────────────

class TestDomainSpeechCorrection(unittest.TestCase):
    def test_correct_domain_phrases_gemini(self):
        from assistant.speech_to_text import correct_domain_phrases
        self.assertEqual(correct_domain_phrases("open gmail"), "open gemini")
        self.assertEqual(correct_domain_phrases("open gem"), "open gemini")
        self.assertEqual(correct_domain_phrases("open jamini"), "open gemini")
        self.assertEqual(correct_domain_phrases("open jemini"), "open gemini")
        self.assertEqual(correct_domain_phrases("open germany"), "open gemini")
        self.assertEqual(correct_domain_phrases("close gmail"), "close gemini")
        self.assertEqual(correct_domain_phrases("close gem"), "close gemini")
        self.assertEqual(correct_domain_phrases("ask gmail who is elon musk"), "ask gemini who is elon musk")
        self.assertEqual(correct_domain_phrases("gem"), "gemini")
        self.assertEqual(correct_domain_phrases("gmail"), "gemini")


# ─── Wake-Word Detector Tests ──────────────────────────────────────────────────

class TestWakeWordDetector(unittest.TestCase):
    def setUp(self):
        from assistant.wake_word import WakeWordDetector
        self.fired_with = []
        self.wwd = WakeWordDetector(on_wake_word=lambda cmd=None: self.fired_with.append(cmd))

    def test_extract_wake_word_pure(self):
        matched, rem = self.wwd._extract_wake_word("wake up")
        self.assertTrue(matched)
        self.assertIsNone(rem)

        matched, rem = self.wwd._extract_wake_word("hey jarvis")
        self.assertTrue(matched)
        self.assertIsNone(rem)

        matched, rem = self.wwd._extract_wake_word("wake up jarvis")
        self.assertTrue(matched)
        self.assertIsNone(rem)

    def test_extract_wake_word_with_command(self):
        matched, rem = self.wwd._extract_wake_word("wake up open gemini")
        self.assertTrue(matched)
        self.assertEqual(rem, "open gemini")

        matched, rem = self.wwd._extract_wake_word("hey jarvis, clear canvas")
        self.assertTrue(matched)
        self.assertEqual(rem, "clear canvas")

    def test_extract_wake_word_negative(self):
        matched, rem = self.wwd._extract_wake_word("hello how are you")
        self.assertFalse(matched)
        self.assertIsNone(rem)

    def test_pause_resume(self):
        self.assertFalse(self.wwd.is_paused)
        self.wwd.pause()
        self.assertTrue(self.wwd.is_paused)
        self.wwd.resume()
        self.assertFalse(self.wwd.is_paused)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
