"""
AeroScript — Gemini AI Chat Engine
====================================
Wraps the Google Gemini API for in-app AI Q&A.

Features:
  - Non-blocking: ask() runs on a background thread, calls callback when done.
  - Keeps answers short (1-2 sentences) so they fit neatly on the canvas.
  - Works without an API key — shows a helpful setup message instead.
  - Maintains a short conversation history for context.
"""

import threading
import logging
import os
from typing import Optional, Callable

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are JARVIS, an AI assistant built into AeroScript, a touchless drawing app. "
    "Answer questions concisely in 1-2 sentences maximum. "
    "If asked to write something, keep it under 15 words. "
    "Be helpful, smart, and friendly."
)

_NO_KEY_MSG  = "No Gemini API key. Set GEMINI_API_KEY env var."
_ERROR_MSG   = "Gemini error. Check your API key or internet connection."


FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


class GeminiChatEngine:
    """Non-blocking Gemini AI chat engine."""

    _KEY_FILE = "config/gemini_key.txt"

    def __init__(self, api_key: str = ""):
        candidate_key   = (api_key or "").strip() or self._load_saved_key()
        self._api_key   = candidate_key if self.is_valid_key(candidate_key) else ""
        self._history   = []
        self._thinking  = False
        self._client    = None
        self._model_name = FALLBACK_MODELS[0]

        if self._api_key:
            self._init_client()

    @staticmethod
    def is_valid_key(key: str) -> bool:
        """
        Validate whether a candidate API key matches expected Gemini API key format.
        Supports standard Google AI Studio keys ('AIza...'), project keys ('AQ...'),
        and generic proxy/token keys.
        """
        if not key or not isinstance(key, str):
            return False
        k = key.strip().strip('"').strip("'")
        # Must not contain whitespace or line breaks
        if any(c.isspace() for c in k):
            return False
        # Filter out common placeholders
        if k.lower() in ("your-key-here", "your_api_key_here", "none", "null", "close gem", "api_key", "your_key"):
            return False
        # Valid key should have sufficient length and valid key characters
        if len(k) >= 20 and all(c.isalnum() or c in "-_." for c in k):
            return True
        return False

    @property
    def has_key(self) -> bool:
        return bool(self._api_key and self.is_valid_key(self._api_key))

    @property
    def thinking(self) -> bool:
        return self._thinking

    @classmethod
    def _load_saved_key(cls) -> str:
        # 1. Environment variable
        env_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
        if cls.is_valid_key(env_key):
            return env_key

        # 2. .env file
        if os.path.isfile(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            k = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if cls.is_valid_key(k):
                                return k
            except Exception:
                pass

        # 3. Local config/gemini_key.txt
        if os.path.isfile(cls._KEY_FILE):
            try:
                with open(cls._KEY_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip().strip('"').strip("'")
                        if cls.is_valid_key(line):
                            return line
            except Exception:
                pass

        return ""

    def set_api_key(self, key: str) -> bool:
        """Set and persist a new API key if valid. Returns True if saved."""
        cleaned = (key or "").strip().strip('"').strip("'")
        if not self.is_valid_key(cleaned):
            log.warning(f"Rejected invalid Gemini API key format: '{cleaned[:10]}...'")
            return False

        self._api_key = cleaned
        try:
            os.makedirs(os.path.dirname(self._KEY_FILE), exist_ok=True)
            with open(self._KEY_FILE, "w", encoding="utf-8") as f:
                f.write(self._api_key + "\n")
            log.info("Gemini API key saved to config/gemini_key.txt")
        except Exception as e:
            log.warning(f"Could not persist API key to file: {e}")

        # Also update .env file if it exists
        try:
            if os.path.isfile(".env"):
                with open(".env", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                new_lines = []
                found = False
                for line in lines:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        new_lines.append(f'GEMINI_API_KEY="{self._api_key}"\n')
                        found = True
                    else:
                        new_lines.append(line)
                if not found:
                    new_lines.append(f'GEMINI_API_KEY="{self._api_key}"\n')
                with open(".env", "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
        except Exception as e:
            log.warning(f"Could not update .env with new key: {e}")

        self._init_client()
        return True

    def ask(self, question: str, callback: Callable[[str, Optional[str]], None]):
        if not question.strip():
            callback("", "Empty question.")
            return
        if not self.has_key:
            callback(_NO_KEY_MSG, None)
            return
        if self._thinking:
            callback("", "Still thinking... please wait.")
            return

        self._thinking = True
        t = threading.Thread(target=self._ask_thread, args=(question, callback),
                             daemon=True, name="gemini-ask")
        t.start()

    def clear_history(self):
        self._history.clear()

    def _init_client(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            log.info(f"Gemini client initialized (model: {self._model_name})")
        except Exception as e:
            log.error(f"Gemini client init failed: {e}")
            self._client = None

    def _ask_thread(self, question: str, callback: Callable[[str, Optional[str]], None]):
        try:
            if not self._client:
                self._init_client()
            if not self._client:
                callback(_ERROR_MSG, "Client not initialized.")
                return

            from google import genai
            # pyrefly: ignore [missing-import]
            from google.genai import types

            contents = list(self._history[-4:])
            contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

            # Build prioritized list starting with the currently preferred model
            models_to_try = [self._model_name] + [m for m in FALLBACK_MODELS if m != self._model_name]
            response = None
            last_err = None

            for m_name in models_to_try:
                try:
                    log.info(f"Querying Gemini (model: {m_name})...")
                    # Build config safely
                    cfg_kwargs = dict(
                        system_instruction=_SYSTEM_PROMPT,
                        max_output_tokens=300,
                        temperature=0.7,
                    )
                    if hasattr(types, "ThinkingConfig"):
                        try:
                            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                        except Exception:
                            pass
                    
                    response = self._client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                    self._model_name = m_name  # Keep the model that succeeded
                    break
                except Exception as ex:
                    last_err = ex
                    err_lower = str(ex).lower()
                    if any(k in err_lower for k in ["429", "quota", "resource_exhausted", "503", "unavailable", "not found"]):
                        log.warning(f"Model {m_name} rate/quota or availability issue: {ex}. Failing over to next model...")
                        continue
                    else:
                        raise

            if response is None:
                if last_err:
                    raise last_err
                callback(_ERROR_MSG, "No response received.")
                return

            answer = ""
            if getattr(response, "text", None):
                answer = response.text.strip()
            elif getattr(response, "candidates", None) and len(response.candidates) > 0:
                candidate = response.candidates[0]
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        pt = getattr(part, "text", None)
                        if pt:
                            answer += pt + " "
                    answer = answer.strip()

            if not answer:
                answer = "(no response)"

            self._history.append(types.Content(role="user",  parts=[types.Part(text=question)]))
            self._history.append(types.Content(role="model", parts=[types.Part(text=answer)]))

            log.info(f"Gemini Q: '{question[:40]}' A: '{answer[:60]}'")
            callback(answer, None)

        except Exception as e:
            log.error(f"Gemini ask error: {e}")
            err_str = str(e)
            if "API_KEY" in err_str.upper() or "401" in err_str:
                callback("Invalid API key. Get one at aistudio.google.com", str(e))
            elif "quota" in err_str.lower() or "429" in err_str:
                callback("Daily free quota reached across models. Try again later or add billing in AI Studio.", str(e))
            else:
                callback(_ERROR_MSG, str(e))
        finally:
            self._thinking = False
