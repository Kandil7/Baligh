"""Tests for chat interface (with mocks)."""

from unittest.mock import MagicMock, patch

from baligh.inference.chat import ChatBot


class TestChatBot:
    @patch("baligh.inference.chat.TextGenerator")
    def test_creation(self, MockGen):
        mock_gen = MagicMock()
        MockGen.return_value = mock_gen
        bot = ChatBot("fake_model")
        assert bot.system_prompt is not None
        assert bot.history == []

    @patch("baligh.inference.chat.TextGenerator")
    def test_chat(self, MockGen):
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "مرحبا"
        MockGen.return_value = mock_gen
        bot = ChatBot("fake_model")
        response = bot.chat("مرحبا")
        assert response == "مرحبا"
        assert len(bot.history) == 2  # user + assistant

    @patch("baligh.inference.chat.TextGenerator")
    def test_clear_history(self, MockGen):
        MockGen.return_value = MagicMock()
        bot = ChatBot("fake_model")
        bot.history = [{"role": "user", "content": "test"}]
        bot.clear_history()
        assert bot.history == []

    @patch("baligh.inference.chat.TextGenerator")
    def test_get_history(self, MockGen):
        MockGen.return_value = MagicMock()
        bot = ChatBot("fake_model")
        bot.history = [{"role": "user", "content": "test"}]
        assert bot.get_history() == [{"role": "user", "content": "test"}]
