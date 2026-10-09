"""
AeroScript 2.0 — AI Features Test Suite
========================================
Tests for:
1. PyTorch CharacterClassifier (43 classes)
2. Safe AST MathEngine
3. AirWritingEngine OCR & Expression Solver
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from ai.character_recognizer import CharacterClassifier, CLASSES
from ai.math_engine import MathEngine
from ai.air_writing import AirWritingEngine


def test_character_classifier_init():
    clf = CharacterClassifier.get_instance()
    assert clf is not None
    assert clf.model is not None


def test_math_engine_arithmetic():
    assert MathEngine.evaluate("2 + 3")["result"] == 5
    assert MathEngine.evaluate("10 - 4")["result"] == 6
    assert MathEngine.evaluate("7 * 8")["result"] == 56
    assert MathEngine.evaluate("7 x 8")["result"] == 56
    assert MathEngine.evaluate("20 / 4")["result"] == 5
    assert MathEngine.evaluate("(2 + 3) * 4")["result"] == 20
    assert MathEngine.evaluate("2 ^ 3")["result"] == 8
    assert MathEngine.evaluate("15 / 2")["result"] == 7.5


def test_math_engine_trailing_equals():
    res = MathEngine.evaluate("2 + 3 =")
    assert res["valid"] is True
    assert res["result"] == 5
    assert res["formatted"] == "2 + 3 = 5"


def test_math_engine_division_by_zero():
    res = MathEngine.evaluate("5 / 0")
    assert res["valid"] is False
    assert "division by zero" in res["error"].lower()


def test_air_writing_math_canvas():
    engine = AirWritingEngine()
    canvas = np.zeros((300, 800, 3), dtype=np.uint8)
    cv2.putText(canvas, "2 + 3", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
    
    result = engine.process_canvas(canvas, mode="math")
    assert result["math_solution"] is not None
    assert result["math_solution"]["valid"] is True
    assert result["math_solution"]["result"] == 5
    assert "2+3 = 5" in result["text"] or "2 + 3 = 5" in result["text"]


def test_air_writing_text_canvas_aero():
    engine = AirWritingEngine()
    canvas = np.zeros((300, 800, 3), dtype=np.uint8)
    cv2.putText(canvas, "AERO", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
    
    result = engine.process_canvas(canvas, mode="text")
    assert result["text"] == "AERO"
    assert result["confidence"] > 0.85
    assert len(result["characters"]) == 4


def test_air_writing_text_canvas_hello():
    engine = AirWritingEngine()
    canvas = np.zeros((300, 800, 3), dtype=np.uint8)
    cv2.putText(canvas, "HELLO", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
    
    result = engine.process_canvas(canvas, mode="text")
    assert result["text"] == "HELLO"
    assert result["confidence"] > 0.85
    assert len(result["characters"]) == 5


if __name__ == "__main__":
    test_character_classifier_init()
    test_math_engine_arithmetic()
    test_math_engine_trailing_equals()
    test_math_engine_division_by_zero()
    test_air_writing_math_canvas()
    test_air_writing_text_canvas_aero()
    test_air_writing_text_canvas_hello()
    print("ALL AI FEATURE TESTS PASSED SUCCESSFULLY!")
