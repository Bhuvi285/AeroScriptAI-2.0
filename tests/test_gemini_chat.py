import unittest
import os
from ai.gemini_chat import GeminiChatEngine


class TestGeminiChatEngine(unittest.TestCase):
    def test_key_validation(self):
        # Valid format (AIza... >= 30 chars)
        valid_mock_key = "AIzaSy" + "A" * 33
        self.assertTrue(GeminiChatEngine.is_valid_key(valid_mock_key))

        # Invalid keys
        self.assertFalse(GeminiChatEngine.is_valid_key(""))
        self.assertFalse(GeminiChatEngine.is_valid_key("close gem"))
        self.assertFalse(GeminiChatEngine.is_valid_key("your-key-here"))
        self.assertFalse(GeminiChatEngine.is_valid_key("short_key"))
        self.assertFalse(GeminiChatEngine.is_valid_key(None))

    def test_init_without_key(self):
        engine = GeminiChatEngine(api_key="invalid key with spaces")
        self.assertFalse(engine.has_key)

    def test_set_invalid_key_rejected(self):
        engine = GeminiChatEngine(api_key="invalid key with spaces")
        saved = engine.set_api_key("close gem")
        self.assertFalse(saved)
        self.assertFalse(engine.has_key)


if __name__ == "__main__":
    unittest.main()
