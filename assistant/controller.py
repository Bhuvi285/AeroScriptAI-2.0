"""
AeroScript JARVIS — AeroScriptController
=========================================
Clean bridge between JARVIS voice commands and the AeroScript App state.

Rules:
  - ONLY called from the OpenCV main thread (after draining the command queue).
  - NEVER called directly from STT/TTS threads.
  - Holds a weak-ish reference to App — checked before use.
  - All methods return bool: True = success, False = nothing to do / error.
"""

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from main import App   # type: ignore  – avoid circular at runtime

log = logging.getLogger(__name__)

# Color name → COLORS list index (matches main.py COLORS order)
_COLOR_NAME_TO_IDX: dict[str, int] = {
    "red":    0,
    "green":  1,
    "blue":   2,
    "yellow": 3,
    "orange": 4,
    "white":  5,
    "black":  6,
}

# Tool name → (tool_idx in TOOLS list, eraser flag)
_TOOL_MAP: dict[str, tuple[int, bool]] = {
    "pen":         (0, False),
    "marker":      (1, False),
    "highlighter": (2, False),
    "eraser":      (0, True),   # eraser is a flag, not a tool slot
}


class AeroScriptController:
    """
    Thin adapter that translates high-level voice commands into
    mutations on the App instance's state.

    All public methods are safe to call when self._app is None
    (they return False and log a warning).
    """

    def __init__(self, app: Optional["App"] = None):
        self._app = app

    def bind(self, app: "App"):
        """Attach to a running App instance."""
        self._app = app
        log.info("AeroScriptController bound to App.")

    # ── Canvas Operations ─────────────────────────────────────────────────────

    def clear_canvas(self) -> bool:
        app = self._app
        if not app or not app.canvas:
            return False
        app.canvas.clear()
        app.ai_result  = None
        app.ai_solved  = False
        app.last_draw_time = 0.0
        app._toast("JARVIS: Canvas cleared")
        log.info("Controller: clear_canvas")
        return True

    def undo(self) -> bool:
        app = self._app
        if app:
            app._toast("JARVIS: Undo is disabled")
        log.info("Controller: undo requested (disabled)")
        return False

    def redo(self) -> bool:
        app = self._app
        if app:
            app._toast("JARVIS: Redo is disabled")
        log.info("Controller: redo requested (disabled)")
        return False

    def save_canvas(self) -> bool:
        app = self._app
        if not app:
            return False
        app._save()
        log.info("Controller: save_canvas")
        return True

    def screenshot(self) -> bool:
        """Save a screenshot of the current full frame (same as save_canvas for now)."""
        return self.save_canvas()

    # ── Color ─────────────────────────────────────────────────────────────────

    def set_color(self, color_name: str = "", color_idx: int = -1) -> bool:
        """
        Set the drawing colour.

        Args:
            color_name: Spoken colour name (e.g. "blue").
            color_idx:  Direct index into COLORS list (0-6). Takes priority.
        """
        app = self._app
        if not app:
            return False

        if color_idx < 0 and color_name:
            color_idx = _COLOR_NAME_TO_IDX.get(color_name.lower(), -1)

        if color_idx < 0 or color_idx >= len(app.COLORS if hasattr(app, "COLORS") else [None]*7):
            log.warning(f"Controller: unknown color '{color_name}' / idx={color_idx}")
            return False

        app.color_idx = color_idx
        app.eraser    = False
        name = color_name or str(color_idx)
        app._toast(f"JARVIS: {name.capitalize()} selected")
        log.info(f"Controller: set_color → idx={color_idx} ({name})")
        return True

    # ── Brush ─────────────────────────────────────────────────────────────────

    def increase_brush(self) -> bool:
        app = self._app
        if not app:
            return False
        app.brush = min(app.brush + 2, 40)
        app._toast(f"JARVIS: Brush {app.brush}")
        log.info(f"Controller: increase_brush → {app.brush}")
        return True

    def decrease_brush(self) -> bool:
        app = self._app
        if not app:
            return False
        app.brush = max(app.brush - 2, 2)
        app._toast(f"JARVIS: Brush {app.brush}")
        log.info(f"Controller: decrease_brush → {app.brush}")
        return True

    # ── Tools ─────────────────────────────────────────────────────────────────

    def select_tool(self, tool_name: str) -> bool:
        """
        Select a drawing tool by name.
        tool_name: 'pen' | 'marker' | 'highlighter' | 'eraser'
        """
        app = self._app
        if not app:
            return False

        tool_name = tool_name.lower()

        if tool_name == "eraser":
            app.eraser = True
            app._toast("JARVIS: Eraser")
            log.info("Controller: select_tool → eraser ON")
            return True

        # Map tool name to TOOLS index
        tool_to_idx = {
            "pen":         0,
            "marker":      1,
            "highlighter": 2,
            "shape":       3,
            "shape_pen":   3,
            "air_text":    4,
            "text":        4,
            "smart_math":  5,
            "math":        5,
        }
        idx = tool_to_idx.get(tool_name, -1)
        if idx < 0:
            log.warning(f"Controller: unknown tool '{tool_name}'")
            return False

        app.tool_idx = idx
        app.eraser   = False
        label = "Smart Math" if tool_name in ("smart_math", "math") else ("Air-Text" if tool_name in ("air_text", "text") else tool_name.capitalize())
        app._toast(f"JARVIS: {label}")
        log.info(f"Controller: select_tool → {tool_name} (idx={idx})")
        return True

    def stamp_result(self) -> bool:
        """Stamp the recognized air-writing, math result, or Gemini answer onto canvas."""
        return self.paste_to_screen()

    def paste_to_screen(self) -> bool:
        """
        Paste the current Gemini answer, math solution, or air-text result
        onto the screen canvas in handwritten format.
        """
        app = self._app
        if not app:
            return False

        # Priority 1: Gemini answer
        if getattr(app, "chat_answer", ""):
            app._stamp_gemini_answer()
            app._toast("JARVIS: Answer pasted on screen!")
            log.info("Controller: paste_to_screen stamped Gemini answer")
            return True

        # Priority 2: Air-text or Smart-math
        if hasattr(app, "_tool") and app._tool() in ("air_text", "smart_math"):
            if app.ai_result and app.ai_result.get("text"):
                app._stamp_ai_result()
                app._toast("JARVIS: Result pasted on screen!")
                log.info("Controller: paste_to_screen stamped AI result")
                return True
            elif hasattr(app, "_run_ai_solve"):
                app._run_ai_solve()
                if app.ai_result and app.ai_result.get("text"):
                    app._stamp_ai_result()
                    app._toast("JARVIS: Result pasted on screen!")
                    return True

        app._toast("JARVIS: Nothing to paste")
        return False

    # ── Modes ─────────────────────────────────────────────────────────────────

    def enable_whiteboard(self) -> bool:
        app = self._app
        if not app:
            return False
        if not app.white_mode:
            app.white_mode = True
            app._toast("JARVIS: Whiteboard ON")
            log.info("Controller: enable_whiteboard")
        return True

    def enable_camera_mode(self) -> bool:
        app = self._app
        if not app:
            return False
        if app.white_mode:
            app.white_mode = False
            app._toast("JARVIS: Camera mode")
            log.info("Controller: enable_camera_mode")
        return True

    def toggle_whiteboard(self) -> bool:
        app = self._app
        if not app:
            return False
        app.white_mode = not app.white_mode
        app._toast("JARVIS: Whiteboard ON" if app.white_mode else "JARVIS: Camera mode")
        return True

    def open_gemini(self) -> bool:
        app = self._app
        if not app:
            return False
        app.chat_mode    = True
        app.chat_input   = ""
        app.chat_answer  = ""
        app.chat_stamped = False
        app._toast("JARVIS: Gemini Chat ON")
        log.info("Controller: open_gemini")
        return True

    def close_gemini(self) -> bool:
        app = self._app
        if not app:
            return False
        app.chat_mode = False
        app._toast("JARVIS: Gemini Chat OFF")
        log.info("Controller: close_gemini")
        return True

    def search_gemini(self, query: str = "") -> bool:
        """
        Populate speech to text into Gemini search box, open Gemini if needed,
        and trigger Gemini query.
        """
        app = self._app
        if not app:
            return False
        q = (query or "").strip()
        if not q:
            app._toast("JARVIS: Empty query")
            return False

        app.chat_mode      = True
        app.chat_input     = q
        app.chat_answer    = ""
        app.chat_stamped   = False
        app.chat_thinking  = True
        app._last_chat_query = q
        disp_q = q[:28] + ("..." if len(q) > 28 else "")
        app._toast(f"JARVIS: Searching '{disp_q}'")
        log.info(f"Controller: search_gemini → '{q}'")

        if hasattr(app, "gemini") and app.gemini:
            app.gemini.ask(q, app._gemini_callback)
        return True

    # ── App Control ───────────────────────────────────────────────────────────

    def close_project(self) -> bool:
        """Close the entire AeroScript project cleanly."""
        app = self._app
        if not app:
            return False
        app.running = False
        app._toast("JARVIS: Closing AeroScript...")
        log.info("Controller: close_project requested")
        return True

    def pause(self) -> bool:
        app = self._app
        if not app:
            return False
        app.jarvis_paused = True
        app._toast("JARVIS: Paused — say 'resume' to continue")
        log.info("Controller: pause")
        return True

    def resume(self) -> bool:
        app = self._app
        if not app:
            return False
        app.jarvis_paused = False
        app._toast("JARVIS: Resumed")
        log.info("Controller: resume")
        return True

    # ── Status / Info ─────────────────────────────────────────────────────────

    def show_status(self) -> bool:
        app = self._app
        if not app or not app.canvas:
            return False
        fps  = getattr(app, '_fps', 0)
        tool_labels = {
            "pen": "PEN", "marker": "MKR", "highlighter": "HLT",
            "shape_pen": "SHP", "air_text": "TXT", "smart_math": "MTH"
        }
        tool = "ERASER" if app.eraser else tool_labels.get(app._tool(), "PEN")
        from main import COLORS  # type: ignore
        color_name = COLORS[app.color_idx][0]
        status = f"Tool:{tool} Color:{color_name} Brush:{app.brush} FPS:{fps}"
        app._toast(status)
        log.info(f"Controller: show_status → {status}")
        return True

    def show_fps(self) -> bool:
        app = self._app
        if not app:
            return False
        fps = getattr(app, '_fps', 0)
        app._toast(f"JARVIS: FPS = {fps}")
        log.info(f"Controller: show_fps → {fps}")
        return True

    # ── Presentation (stub — extend as needed) ────────────────────────────────

    def start_presentation(self) -> bool:
        app = self._app
        if not app:
            return False
        app._toast("JARVIS: Presentation mode")
        log.info("Controller: start_presentation (stub)")
        return True

    def stop_presentation(self) -> bool:
        app = self._app
        if not app:
            return False
        app._toast("JARVIS: Presentation stopped")
        log.info("Controller: stop_presentation (stub)")
        return True

    def start_annotation(self) -> bool:
        app = self._app
        if not app:
            return False
        app._toast("JARVIS: Annotation ON")
        return True

    def stop_annotation(self) -> bool:
        app = self._app
        if not app:
            return False
        app._toast("JARVIS: Annotation OFF")
        return True
