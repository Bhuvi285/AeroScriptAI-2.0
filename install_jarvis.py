#!/usr/bin/env python3
"""
====================================================================
  AeroScript — JARVIS Voice Assistant Installation & Verification
====================================================================
Run this script to verify that all JARVIS dependencies are installed
and working correctly BEFORE launching AeroScript.

Usage:
    python3 install_jarvis.py

What it checks:
  1. Python version (3.9+)
  2. Virtual environment status
  3. SpeechRecognition library
  4. PyAudio (microphone input)
  5. pyttsx3 (text-to-speech)
  6. JARVIS assistant modules
  7. Microphone hardware availability
  8. TTS engine initialisation
  9. Command processor (parse test)
 10. AeroScript core modules

Exit code: 0 = all critical checks pass, 1 = critical failure.
====================================================================
"""

import sys
import os
import subprocess
import importlib
import time

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

OK      = f"{GREEN}✓{RESET}"
FAIL    = f"{RED}✗{RESET}"
WARN    = f"{YELLOW}⚠{RESET}"
SKIP    = f"{DIM}–{RESET}"


def header(text: str):
    print(f"\n{BOLD}{CYAN}{text}{RESET}")
    print("─" * 60)


def row(label: str, status: str, detail: str = ""):
    pad = max(0, 36 - len(label))
    line = f"  {label}{' ' * pad}{status}"
    if detail:
        line += f"  {DIM}{detail}{RESET}"
    print(line)


def pip_install(package: str) -> bool:
    """Attempt to pip-install a package. Returns True on success."""
    print(f"  → Installing {package}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"{GREEN}done{RESET}")
        return True
    else:
        print(f"{RED}failed{RESET}")
        return False


# ── Individual checks ─────────────────────────────────────────────────────────

def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 9)
    ver = f"{v.major}.{v.minor}.{v.micro}"
    row("Python", OK if ok else FAIL, ver)
    return ok


def check_venv() -> None:
    in_venv = (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    row("Virtual environment", OK if in_venv else WARN,
        "active" if in_venv else "not detected — consider using a venv")


def check_package(import_name: str, display_name: str,
                  required: bool = True, install_name: str = "") -> bool:
    """Try to import a package; optionally install if missing."""
    try:
        importlib.import_module(import_name)
        # Try to get version
        try:
            import importlib.metadata as meta
            ver = meta.version(install_name or import_name)
        except Exception:
            ver = ""
        row(display_name, OK, ver)
        return True
    except ImportError:
        if required and install_name:
            row(display_name, WARN, f"not found — attempting install")
            if pip_install(install_name):
                return check_package(import_name, display_name, required, install_name)
        row(display_name, FAIL if required else SKIP,
            "not installed" if required else "optional")
        return not required


def check_microphone() -> bool:
    """Check that at least one microphone device is accessible."""
    try:
        import speech_recognition as sr
        import pyaudio
        pa = pyaudio.PyAudio()
        count = pa.get_device_count()
        input_devs = [
            pa.get_device_info_by_index(i)
            for i in range(count)
            if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
        ]
        pa.terminate()
        if input_devs:
            name = input_devs[0].get("name", "unknown")[:32]
            row("Microphone hardware", OK, f"{len(input_devs)} device(s) — '{name}'")
            return True
        else:
            row("Microphone hardware", FAIL, "no input devices found")
            return False
    except Exception as e:
        row("Microphone hardware", WARN, f"check failed: {e}")
        return False


def check_mic_calibration() -> bool:
    """Quick ambient-noise calibration (0.5 s)."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as src:
            r.adjust_for_ambient_noise(src, duration=0.5)
        row("Ambient noise calibration", OK,
            f"energy threshold = {r.energy_threshold:.0f}")
        return True
    except Exception as e:
        row("Ambient noise calibration", WARN, str(e)[:50])
        return False


def check_tts() -> bool:
    """Initialise pyttsx3 engine and speak a silent test."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("volume", 0.0)  # silent
        engine.say("test")
        engine.runAndWait()
        engine.stop()
        row("Text-to-speech engine", OK)
        return True
    except Exception as e:
        row("Text-to-speech engine", FAIL, str(e)[:50])
        return False


def check_jarvis_modules() -> bool:
    """Verify all JARVIS modules import cleanly."""
    modules = [
        ("assistant.responses",       "assistant.responses"),
        ("assistant.command_processor","assistant.command_processor"),
        ("assistant.speech_to_text",  "assistant.speech_to_text"),
        ("assistant.text_to_speech",  "assistant.text_to_speech"),
        ("assistant.wake_word",       "assistant.wake_word"),
        ("assistant.controller",      "assistant.controller"),
        ("assistant.jarvis",          "assistant.jarvis"),
    ]
    all_ok = True
    for mod, label in modules:
        try:
            importlib.import_module(mod)
            row(label, OK)
        except Exception as e:
            row(label, FAIL, str(e)[:50])
            all_ok = False
    return all_ok


def check_command_processor() -> bool:
    """Run a set of parse assertions."""
    try:
        from assistant.command_processor import CommandProcessor, Intent
        cp = CommandProcessor()
        tests = [
            ("clear the canvas",        Intent.CLEAR_CANVAS),
            ("undo",                    Intent.UNDO),
            ("change color to blue",    Intent.CHANGE_COLOR),
            ("select eraser",           Intent.SELECT_ERASER),
            ("enable whiteboard mode",  Intent.ENABLE_WHITEBOARD),
            ("save the canvas",         Intent.SAVE_CANVAS),
            ("increase brush",          Intent.INCREASE_BRUSH),
            ("xyzzy abracadabra",       Intent.UNKNOWN),
        ]
        failures = []
        for phrase, expected in tests:
            cmd = cp.process(phrase)
            if cmd.intent != expected:
                failures.append(f"'{phrase}' → {cmd.intent.name} (expected {expected.name})")
        if failures:
            row("Command processor", FAIL, "; ".join(failures[:2]))
            return False
        row("Command processor", OK, f"{len(tests)} phrases parsed correctly")
        return True
    except Exception as e:
        row("Command processor", FAIL, str(e)[:60])
        return False


def check_aeroscript_core() -> bool:
    """Verify AeroScript core modules import OK."""
    mods = [
        ("cv2",               "OpenCV"),
        ("mediapipe",         "MediaPipe"),
        ("numpy",             "NumPy"),
        ("core.canvas",       "core.canvas"),
        ("core.drawing_engine","core.drawing_engine"),
    ]
    all_ok = True
    for mod, label in mods:
        try:
            importlib.import_module(mod)
            row(label, OK)
        except Exception as e:
            row(label, WARN, str(e)[:50])
            # Not fatal for JARVIS install
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print(f"{BOLD}{'=' * 60}")
    print(f"  AEROSCRIPT — JARVIS INSTALLATION CHECK")
    print(f"{'=' * 60}{RESET}")

    # Ensure we run from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    results: dict[str, bool] = {}

    # 1. Python
    header("1. Python Environment")
    results["Python version"]   = check_python()
    check_venv()

    if not results["Python version"]:
        print(f"\n{RED}Python 3.9+ is required. Please upgrade.{RESET}")
        return 1

    # 2. Voice dependencies
    header("2. Voice Dependencies")
    results["SpeechRecognition"] = check_package(
        "speech_recognition", "SpeechRecognition", True, "SpeechRecognition")
    results["PyAudio"]           = check_package(
        "pyaudio", "PyAudio", True, "PyAudio")
    results["pyttsx3"]           = check_package(
        "pyttsx3", "pyttsx3", True, "pyttsx3")

    # 3. JARVIS modules
    header("3. JARVIS Modules")
    results["JARVIS modules"] = check_jarvis_modules()

    # 4. Command processor
    header("4. Command Processor")
    results["Command processor"] = check_command_processor()

    # 5. Audio hardware
    header("5. Audio Hardware")
    results["Microphone"] = check_microphone()
    if results["Microphone"]:
        results["Calibration"] = check_mic_calibration()

    # 6. TTS
    header("6. Text-to-Speech")
    results["TTS"] = check_tts()

    # 7. AeroScript core
    header("7. AeroScript Core")
    check_aeroscript_core()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"{BOLD}{'=' * 60}")
    print(f"  INSTALLATION SUMMARY")
    print(f"{'=' * 60}{RESET}")

    critical = ["Python version", "SpeechRecognition", "PyAudio",
                "pyttsx3", "JARVIS modules", "Command processor"]
    all_critical_ok = all(results.get(k, False) for k in critical)

    for label, ok in results.items():
        row(label, OK if ok else FAIL)

    print()
    if all_critical_ok:
        print(f"{BOLD}{GREEN}✓ JARVIS INSTALLATION SUCCESSFUL{RESET}")
        print(f"{DIM}  Press V in AeroScript to start push-to-talk listening.{RESET}")
        if not results.get("Microphone"):
            print(f"{YELLOW}  ⚠ No microphone detected — voice input may not work.{RESET}")
        return 0
    else:
        print(f"{BOLD}{RED}✗ JARVIS INSTALLATION INCOMPLETE{RESET}")
        failed = [k for k in critical if not results.get(k, False)]
        print(f"{RED}  Failed: {', '.join(failed)}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
