"""Tests for tokenizer utilities."""

from unittest.mock import MagicMock, patch

from baligh.models.tokenizer import (
    count_tokens,
    format_instruction,
    get_chat_template,
    get_tokenizer,
)


class TestGetTokenizer:
    """Tests for get_tokenizer."""

    @patch("baligh.models.tokenizer.AutoTokenizer")
    def test_default_params(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = get_tokenizer()

        mock_auto_tokenizer.from_pretrained.assert_called_once()
        assert tokenizer.padding_side == "right"
        assert tokenizer.truncation_side == "right"

    @patch("baligh.models.tokenizer.AutoTokenizer")
    def test_custom_params(self, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "eos"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = get_tokenizer(
            tokenizer_name="custom/model", max_seq_length=4096, padding_side="left"
        )

        assert tokenizer.padding_side == "left"
        assert tokenizer.model_max_length == 4096


class TestGetChatTemplate:
    """Tests for get_chat_template."""

    def test_returns_string(self):
        mock_tokenizer = MagicMock()
        template = get_chat_template(mock_tokenizer)
        assert isinstance(template, str)

    def test_contains_roles(self):
        mock_tokenizer = MagicMock()
        template = get_chat_template(mock_tokenizer)
        assert "system" in template
        assert "user" in template
        assert "assistant" in template

    def test_uses_valid_jinja_syntax(self):
        """Regression guard: v0.1 shipped {{% ... %}} which crashes at render."""
        from jinja2 import Environment, TemplateSyntaxError

        template = get_chat_template(MagicMock())
        try:
            Environment().parse(template)
        except TemplateSyntaxError as exc:
            raise AssertionError(f"Fallback chat template is invalid Jinja: {exc}") from exc

    def test_template_renders_chatml(self):
        from jinja2 import Environment

        template = get_chat_template(MagicMock())
        rendered = (
            Environment()
            .from_string(template)
            .render(
                messages=[
                    {"role": "system", "content": "أنت مساعد"},
                    {"role": "user", "content": "مرحبا"},
                    {"role": "assistant", "content": "أهلاً"},
                ],
                add_generation_prompt=False,
            )
        )
        assert "<|im_start|>user\nمرحبا<|im_end|>" in rendered
        assert "<|im_start|>assistant\nأهلاً<|im_end|>" in rendered

    def test_generation_prompt_appended(self):
        from jinja2 import Environment

        template = get_chat_template(MagicMock())
        rendered = (
            Environment()
            .from_string(template)
            .render(
                messages=[{"role": "user", "content": "سؤال"}],
                add_generation_prompt=True,
            )
        )
        assert rendered.endswith("<|im_start|>assistant\n")


class TestFormatInstruction:
    """Tests for format_instruction."""

    def test_instruction_only(self):
        messages = format_instruction("اكتب قصيدة")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "اكتب قصيدة"

    def test_instruction_with_input(self):
        messages = format_instruction("ترجم", "Hello")
        assert len(messages) == 1
        assert "ترجم" in messages[0]["content"]
        assert "Hello" in messages[0]["content"]

    def test_instruction_with_output(self):
        messages = format_instruction("اكتب", output="جواب")
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "جواب"


class TestCountTokens:
    """Tests for count_tokens."""

    @patch("baligh.models.tokenizer.get_tokenizer")
    def test_count(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_get_tokenizer.return_value = mock_tokenizer

        count = count_tokens(mock_tokenizer, "Hello world")
        assert count == 5
