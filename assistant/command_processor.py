"""
AeroScript JARVIS — Command Processor
======================================
Converts raw speech text → Intent + parameters using a declarative
keyword-pattern table. No giant if/elif chains.

Architecture:
    raw text
       ↓
    normalize  (lowercase, strip punctuation, remove filler words)
       ↓
    iterate INTENT_MAP  (priority-ordered list of IntentRule)
       ↓
    first match → VoiceCommand(intent, params)
       ↓
    AeroScriptController.execute(cmd)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .latency_tracker import LatencyTracker

from .responses import Intent

log = logging.getLogger(__name__)

# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class VoiceCommand:
    """A parsed voice command ready for the controller to execute."""
    intent: Intent
    params: dict = field(default_factory=dict)
    raw_text: str = ""
    _tracker: Optional["LatencyTracker"] = field(default=None, repr=False)

    def __repr__(self):
        return f"VoiceCommand({self.intent.name}, params={self.params})"


@dataclass
class IntentRule:
    """
    Declarative rule binding a set of keyword patterns to an Intent.

    Fields:
        intent      – the Intent to fire when matched
        keywords    – list of keyword-phrases; ANY must appear in the text
        extractor   – optional function(text) → dict of extra params
        require_all – if True, ALL keywords must appear (AND logic)
    """
    intent: Intent
    keywords: list[str]
    extractor: Optional[Callable[[str], dict]] = None
    require_all: bool = False


# ── Color helpers ─────────────────────────────────────────────────────────────

# Maps spoken colour names → (colour index in main.py COLORS list, display name)
_COLOR_MAP: dict[str, tuple[int, str]] = {
    # Primary
    "red":    (0, "Red"),
    "green":  (1, "Green"),
    "blue":   (2, "Blue"),
    "yellow": (3, "Yellow"),
    "orange": (4, "Orange"),
    "white":  (5, "White"),
    "black":  (6, "Black"),
    # Aliases
    "scarlet":  (0, "Red"),
    "crimson":  (0, "Red"),
    "lime":     (1, "Green"),
    "navy":     (2, "Blue"),
    "azure":    (2, "Blue"),
    "sky blue": (2, "Blue"),
    "gold":     (3, "Yellow"),
    "amber":    (4, "Orange"),
    "grey":     (6, "Black"),
    "gray":     (6, "Black"),
    "dark":     (6, "Black"),
}

# Sorted longest-first so "sky blue" beats "blue"
_COLOR_PATTERNS = sorted(_COLOR_MAP.keys(), key=len, reverse=True)


def _extract_color(text: str, raw_text: str = "") -> dict:
    """Extract colour index and name from text. Returns {} if none found."""
    for pat in _COLOR_PATTERNS:
        if re.search(r"\b" + re.escape(pat) + r"\b", text):
            idx, name = _COLOR_MAP[pat]
            return {"color_idx": idx, "color_name": name}
    return {}


def _extract_gemini_query(text: str, raw_text: str = "") -> dict:
    """Extract search/ask query for Gemini from text or raw_text."""
    source = raw_text.strip() if raw_text.strip() else text.strip()
    # Strip common assistant filler / vocatives
    source = re.sub(
        r"^(?:hey\s+jarvis|ok\s+jarvis|okay\s+jarvis|jarvis|please|can\s+you|could\s+you)\s*[,:]?\s*",
        "", source, flags=re.IGNORECASE
    ).strip()

    # Match search / query prefixes
    m = re.search(
        r"^(?:search\s+for|search\s+in\s+gemini|search\s+on\s+gemini|search|ask\s+gemini\s+about|ask\s+gemini\s+to|ask\s+gemini|ask\s+about|ask|query|find\s+out|find)\s+(.+)$",
        source, re.IGNORECASE
    )
    if m:
        query = m.group(1).strip()
        query = re.sub(r"[?!.]+$", "", query).strip()
        if query:
            return {"query": query}

    # Fallback to normalized text match if source didn't start with prefix
    m_norm = re.search(r"\b(?:search|ask|query|find)\s+(.+)$", text, re.IGNORECASE)
    if m_norm:
        q = m_norm.group(1).strip()
        if q:
            return {"query": q}

    return {"query": source}


# ── Normalisation ─────────────────────────────────────────────────────────────

_FILLER_RE = re.compile(
    r"\b(please|can you|could you|jarvis|hey|ok|okay|the|a|an|my|some|just|now)\b"
)
_PUNCT_RE = re.compile(r"[^\w\s]")

_HOMOPHONES = {
    "read": "red",
    "write": "white",
    "blew": "blue",
    "too": "2",
    "to": "2",
    "for": "4",
    "clean": "clear",
    "gmail": "gemini",
    "jamini": "gemini",
    "jemini": "gemini",
    "jini": "gemini",
}

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation, strip filler words, fix homophones."""
    t = text.lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    t = _FILLER_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    words = t.split()
    corrected = [_HOMOPHONES.get(w, w) for w in words]
    return " ".join(corrected)


# ── Intent Map ────────────────────────────────────────────────────────────────
# Order matters — more specific rules first.

INTENT_MAP: list[IntentRule] = [

    # ── Application Shutdown / Exit ───────────────────────────────────────────
    IntentRule(
        intent=Intent.CLOSE_PROJECT,
        keywords=[
            "close aeroscript", "close the aeroscript", "close aero script",
            "close the aero script", "close project", "close the project",
            "exit aeroscript", "quit aeroscript", "exit project", "quit project",
            "shutdown aeroscript", "terminate aeroscript", "close app",
            "exit app", "quit app", "close application",
        ],
    ),

    # ── Gemini Specific Voice Commands ────────────────────────────────────────
    IntentRule(
        intent=Intent.CLOSE_GEMINI,
        keywords=[
            "close the gemini", "close gemini", "exit gemini", "stop gemini",
            "close chat", "exit chat", "stop chat", "leave gemini",
            "dismiss gemini", "hide gemini", "shut gemini",
            "close gem", "close the gem", "close jamini", "close jini",
            "close gmail", "close germany", "exit gmail", "stop gmail",
        ],
    ),
    IntentRule(
        intent=Intent.SEARCH_GEMINI,
        keywords=[
            "search", "search for", "search in gemini", "search on gemini",
            "ask gemini", "ask", "query", "find",
            "ask gmail", "ask gem", "search in gmail", "search gmail",
        ],
        extractor=_extract_gemini_query,
    ),
    IntentRule(
        intent=Intent.PASTE_TO_SCREEN,
        keywords=[
            "paste into screen", "paste in to screen", "paste to screen",
            "paste on screen", "paste on the screen", "paste in screen",
            "paste in the screen", "paste on canvas", "paste to canvas",
            "paste into canvas", "paste in canvas", "paste answer",
            "paste result", "paste it", "paste", "stamp on screen",
            "stamp in screen",
        ],
    ),
    IntentRule(
        intent=Intent.OPEN_GEMINI,
        keywords=[
            "open gemini", "gemini", "start gemini", "enable gemini", "gemini chat",
            "open chat", "start chat", "ai chat", "open ai", "launch gemini",
            "open gem", "open gmail", "open jamini", "open jemini", "open jini",
            "open germany", "open jimmy", "open g mini", "gem", "gmail",
            "start gem", "start gmail", "launch gem", "launch gmail",
        ],
    ),

    # ── Canvas ────────────────────────────────────────────────────────────────
    IntentRule(
        intent=Intent.CLEAR_CANVAS,
        keywords=[
            "clear", "clear canvas", "clear everything", "clear all", "clean canvas",
            "erase canvas", "erase everything", "wipe canvas", "reset canvas",
            "clear it", "empty canvas",
            # Extra natural phrases
            "clear the canvas", "clear the screen", "clear screen",
            "wipe the canvas", "wipe it", "wipe everything",
            "erase all", "erase the canvas", "erase the board",
            "clean everything", "clean it", "delete everything",
            "start fresh", "start over", "new canvas",
            "remove everything", "remove everything from the canvas", "delete all",
        ],
    ),
    IntentRule(
        intent=Intent.UNDO,
        keywords=["undo", "go back", "revert", "take that back", "step back"],
    ),
    IntentRule(
        intent=Intent.REDO,
        keywords=["redo", "go forward", "redo that", "repeat that"],
    ),
    IntentRule(
        intent=Intent.SAVE_CANVAS,
        keywords=["save canvas", "save drawing", "save image", "save it",
                  "save the drawing", "export canvas", "export image", "save"],
    ),
    IntentRule(
        intent=Intent.SCREENSHOT,
        keywords=["screenshot", "take screenshot", "capture screen", "snap",
                  "take a picture", "take picture"],
    ),

    # ── Modes (Whiteboard & Camera) ───────────────────────────────────────────
    IntentRule(
        intent=Intent.ENABLE_WHITEBOARD,
        keywords=[
            "whiteboard", "white board", "whiteboard mode", "enable whiteboard",
            "open whiteboard", "open white board", "open whiteboard mode",
            "switch to whiteboard", "start whiteboard", "white background", "white mode",
        ],
    ),
    IntentRule(
        intent=Intent.ENABLE_CAMERA_MODE,
        keywords=[
            "camera mode", "camera view", "enable camera", "live camera",
            "close whiteboard", "camera background", "camera", "exit whiteboard",
            "disable whiteboard", "turn off whiteboard",
        ],
    ),

    # ── Color ─────────────────────────────────────────────────────
    IntentRule(
        intent=Intent.CHANGE_COLOR,
        keywords=[
            # Direct bare color words
            "red", "green", "blue", "yellow", "orange", "white", "black",
            "scarlet", "crimson", "lime", "navy", "azure", "sky blue", "gold", "amber", "grey", "gray", "dark",
            # color noun phrases
            "red color", "green color", "blue color", "yellow color", "orange color", "white color", "black color",
            "color red", "color green", "color blue", "color yellow", "color orange", "color white", "color black",
            "red pen", "green pen", "blue pen", "yellow pen", "orange pen", "white pen", "black pen",
            # Explicit phrases
            "color", "colour",
            # verb + color patterns
            "use red", "use blue", "use green", "use yellow",
            "use orange", "use white", "use black",
            "make it red", "make it blue", "make it green",
            "make it yellow", "make it orange", "make it white", "make it black",
            "switch to red", "switch to blue", "switch to green",
            "switch to yellow", "switch to orange", "switch to white", "switch to black",
            "change to red", "change to blue", "change to green",
            "change to yellow", "change to orange", "change to white", "change to black",
            "change color to red", "change color to blue", "change color to green",
            "change color to yellow", "change color to orange", "change color to white", "change color to black",
            "turn to red", "turn to blue", "turn to green",
            "turn red", "turn blue", "turn green",
            "set to red", "set to blue", "set to green",
            "select red", "select blue", "select green",
            "select yellow", "select orange", "select white", "select black",
        ],
        extractor=_extract_color,
    ),

    # ── Brush ─────────────────────────────────────────────────────────────────
    IntentRule(
        intent=Intent.INCREASE_BRUSH,
        keywords=["increase brush", "bigger brush", "larger brush", "brush up",
                  "make brush bigger", "thicker", "increase size", "bigger"],
    ),
    IntentRule(
        intent=Intent.DECREASE_BRUSH,
        keywords=["decrease brush", "smaller brush", "brush down", "reduce brush",
                  "make brush smaller", "thinner", "decrease size", "smaller"],
    ),

    # ── Stamp AI Result ───────────────────────────────────────────────────────
    IntentRule(
        intent=Intent.STAMP_RESULT,
        keywords=["stamp", "stamp result", "stamp answer", "stamp it",
                  "print result", "print answer", "print on screen", "print it",
                  "write result", "write answer"],
    ),

    # ── Tools ─────────────────────────────────────────────────────────────────
    IntentRule(
        intent=Intent.SELECT_ERASER,
        keywords=["eraser", "select eraser", "use eraser", "enable eraser",
                  "switch to eraser", "erase mode"],
    ),
    IntentRule(
        intent=Intent.SELECT_HIGHLIGHTER,
        keywords=["highlighter", "select highlighter", "use highlighter",
                  "highlight mode", "switch to highlighter"],
    ),
    IntentRule(
        intent=Intent.SELECT_MARKER,
        keywords=["marker", "select marker", "use marker", "marker mode",
                  "switch to marker"],
    ),
    IntentRule(
        intent=Intent.SELECT_PEN,
        keywords=["pen", "select pen", "use pen", "drawing mode",
                  "switch to pen", "pencil", "draw mode", "back to pen"],
    ),

    # ── Modes (Smart Math & Air Text) ─────────────────────────────────────────
    IntentRule(
        intent=Intent.EXIT_MATH,
        keywords=[
            "exit math", "exit smart math", "stop math", "close math",
            "close math mode", "exit math mode", "disable math", "leave math",
        ],
    ),
    IntentRule(
        intent=Intent.ENABLE_MATH,
        keywords=[
            "open math mode", "open math", "math mode", "smart math", "math solver",
            "enable math", "start math", "math", "enter math", "switch to math",
            "turn on math",
        ],
    ),
    IntentRule(
        intent=Intent.EXIT_AIR_TEXT,
        keywords=["exit air text", "exit text", "stop air writing", "stop text",
                  "close text", "disable text", "leave air text"],
    ),
    IntentRule(
        intent=Intent.ENABLE_AIR_TEXT,
        keywords=["air text", "air writing", "text mode", "enable air text",
                  "start air writing", "write text", "open air text"],
    ),
    IntentRule(
        intent=Intent.START_PRESENTATION,
        keywords=["presentation mode", "start presentation", "present",
                  "begin presentation", "enable presentation"],
    ),
    IntentRule(
        intent=Intent.STOP_PRESENTATION,
        keywords=["stop presentation", "end presentation", "exit presentation",
                  "stop presenting"],
    ),
    IntentRule(
        intent=Intent.START_ANNOTATION,
        keywords=["start annotation", "annotation mode", "enable annotation",
                  "annotate", "begin annotation"],
    ),
    IntentRule(
        intent=Intent.STOP_ANNOTATION,
        keywords=["stop annotation", "end annotation", "exit annotation",
                  "disable annotation"],
    ),

    # ── App Control ─────────────────────────────────────────────────
    # RESUME must be before PAUSE so "unpause" doesn't match "pause" first
    IntentRule(
        intent=Intent.RESUME,
        keywords=[
            "resume", "continue", "unpause", "go", "start again", "carry on",
            "wake up", "wake up jarvis", "wake up aeroscript", "wake",
        ],
    ),
    IntentRule(
        intent=Intent.PAUSE,
        keywords=["pause", "freeze", "stop drawing", "hold on", "wait"],
    ),

    # ── Info ──────────────────────────────────────────────────────────────────
    IntentRule(
        intent=Intent.SHOW_FPS,
        keywords=["fps", "frame rate", "frames per second", "show fps",
                  "what is fps", "current fps"],
    ),
    IntentRule(
        intent=Intent.SHOW_STATUS,
        keywords=["status", "show status", "what is the status", "system status",
                  "info", "show info"],
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────

class CommandProcessor:
    """
    Converts raw speech text into a VoiceCommand using the INTENT_MAP table.

    Usage:
        processor = CommandProcessor()
        cmd = processor.process("Jarvis, change color to blue")
        # VoiceCommand(CHANGE_COLOR, params={"color_idx": 2, "color_name": "Blue"})
    """

    def process(self, raw_text: str, chat_mode_active: bool = False) -> VoiceCommand:
        """
        Parse raw speech text and return a VoiceCommand.

        Args:
            raw_text: The raw string returned by the STT engine.
            chat_mode_active: True if Gemini chat overlay is currently open.

        Returns:
            VoiceCommand with intent and params set. Falls back to
            Intent.SEARCH_GEMINI if chat_mode_active, or Intent.UNKNOWN.
        """
        normalized = _normalize(raw_text)
        log.debug(f"Normalized: '{normalized}' (chat_mode={chat_mode_active})")

        for rule in INTENT_MAP:
            matched = self._matches(normalized, rule)
            if matched:
                if rule.extractor:
                    try:
                        params = rule.extractor(normalized, raw_text)
                    except TypeError:
                        params = rule.extractor(normalized)
                    
                    # SAFETY CHECK: If extractor failed to find required parameters (e.g. color), abort.
                    if not params:
                        log.warning(f"Matched {rule.intent.name} but missing parameters. Falling back to UNKNOWN.")
                        return VoiceCommand(intent=Intent.UNKNOWN, raw_text=raw_text)
                else:
                    params = {}
                cmd = VoiceCommand(intent=rule.intent, params=params, raw_text=raw_text)
                log.info(f"Intent: {rule.intent.name} | text='{raw_text}' | params={params}")
                return cmd

        # If Gemini chat overlay is open and user speaks any question, route it to Gemini!
        if chat_mode_active and raw_text.strip():
            clean_q = re.sub(
                r"^(?:hey\s+jarvis|ok\s+jarvis|okay\s+jarvis|jarvis|please)\s*[,:]?\s*",
                "", raw_text.strip(), flags=re.IGNORECASE
            ).strip()
            if clean_q:
                log.info(f"Chat mode active: routing speech to Gemini search: '{clean_q}'")
                return VoiceCommand(intent=Intent.SEARCH_GEMINI, params={"query": clean_q}, raw_text=raw_text)

        log.info(f"No intent match for: '{raw_text}'")
        return VoiceCommand(intent=Intent.UNKNOWN, raw_text=raw_text)

    @staticmethod
    def _matches(text: str, rule: IntentRule) -> bool:
        """Check if normalized text satisfies the rule's keyword conditions."""
        if rule.require_all:
            return all(bool(re.search(r"\b" + re.escape(kw) + r"\b", text)) for kw in rule.keywords)
        else:
            return any(bool(re.search(r"\b" + re.escape(kw) + r"\b", text)) for kw in rule.keywords)
