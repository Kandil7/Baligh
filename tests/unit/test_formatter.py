"""Tests for prompt formatting — including real completion-masking behavior."""

from unittest.mock import MagicMock, patch

from baligh.data.formatter import PromptFormatter, get_cpt_formatter, get_sft_formatter

IGNORE_INDEX = -100


class FakeTokenizer:
    """Deterministic whitespace tokenizer for behavior tests.

    Token id = stable small int per word. Implements enough of the HF
    tokenizer surface (chat_template + apply_chat_template) so the REAL
    baligh.models.tokenizer.apply_chat_template integration path is
    exercised end-to-end.
    """

    chat_template = None

    def __init__(self):
        self.vocab = {}
        self.model_max_length = 2048

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = 100 + len(self.vocab)
        return self.vocab[token]

    def __call__(self, text, truncation=False, max_length=None, padding=False, return_tensors=None):
        tokens = text.replace("<|im_start|>", " <S> ").replace("<|im_end|>", " <E> ").split()
        if truncation and max_length:
            tokens = tokens[:max_length]
        ids = [self._id(t) for t in tokens]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        if tokenize:
            return self(text)["input_ids"]
        return text


def make_formatter(**kwargs) -> PromptFormatter:
    with patch("baligh.data.formatter.get_tokenizer", return_value=FakeTokenizer()):
        return PromptFormatter(max_seq_length=2048, **kwargs)


class TestPromptFormatterInit:
    @patch("baligh.data.formatter.get_tokenizer")
    def test_init(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = MagicMock()
        formatter = PromptFormatter(max_seq_length=1024)
        assert formatter.max_seq_length == 1024
        assert formatter.packing is False
        assert formatter.response_only_loss is True


class TestFormatCPT:
    @patch("baligh.data.formatter.get_tokenizer")
    def test_format_cpt(self, mock_get_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
        }
        mock_get_tokenizer.return_value = mock_tokenizer

        formatter = PromptFormatter()
        result = formatter.format_cpt({"text": "نص تجريبي"})

        assert result["labels"] == result["input_ids"]
        assert len(result["labels"]) == 5

    @patch("baligh.data.formatter.get_tokenizer")
    def test_format_cpt_empty_text(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = MagicMock()
        formatter = PromptFormatter()
        result = formatter.format_cpt({"text": ""})
        assert result["input_ids"] == []
        assert result["labels"] == []


class TestFormatSFTMasking:
    """Behavior tests for response-only loss masking (the v0.2 fix)."""

    def test_prompt_tokens_masked_response_kept(self):
        formatter = make_formatter(response_only_loss=True)
        example = {
            "instruction": "اكتب كلمة",
            "input": "",
            "output": "مرحبا",
        }
        result = formatter.format_sft(example)

        input_ids = result["input_ids"]
        labels = result["labels"]

        assert len(input_ids) == len(labels) > 0
        # Prompt prefix must be fully masked...
        n_masked = sum(1 for x in labels if x == IGNORE_INDEX)
        assert n_masked > 0
        # ...and the first masked span must align with the unmasked suffix.
        first_real = labels.index(next(x for x in labels if x != IGNORE_INDEX))
        assert all(x == IGNORE_INDEX for x in labels[:first_real])
        # Unmasked labels must equal the corresponding input ids.
        assert labels[first_real:] == input_ids[first_real:]

    def test_full_sequence_loss_when_disabled(self):
        formatter = make_formatter(response_only_loss=False)
        result = formatter.format_sft({"instruction": "اكتب كلمة", "input": "", "output": "مرحبا"})
        assert all(x != IGNORE_INDEX for x in result["labels"])
        assert result["labels"] == result["input_ids"]

    def test_missing_fields_return_empty(self):
        formatter = make_formatter()
        assert formatter.format_sft({"instruction": "", "output": "x"})["input_ids"] == []
        assert formatter.format_sft({"instruction": "x", "output": ""})["input_ids"] == []

    def test_batch_dispatch_matches_single_example(self):
        formatter = make_formatter()
        batch = {
            "instruction": ["سؤال"],
            "input": [""],
            "output": ["جواب"],
        }
        batched = formatter(batch)
        single = formatter.format_sft({"instruction": "سؤال", "input": "", "output": "جواب"})
        assert batched["input_ids"] == [single["input_ids"]]
        assert batched["labels"] == [single["labels"]]

    def test_cpt_batch(self):
        formatter = make_formatter()
        result = formatter({"text": ["نص أول", "نص ثاني"]})
        assert len(result["input_ids"]) == 2
        for ids, labels in zip(result["input_ids"], result["labels"], strict=True):
            assert labels == ids


class TestFactoryFunctions:
    @patch("baligh.data.formatter.get_tokenizer")
    def test_get_cpt_formatter(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = MagicMock()
        formatter = get_cpt_formatter()
        assert formatter.packing is True

    @patch("baligh.data.formatter.get_tokenizer")
    def test_get_sft_formatter_masks_by_default(self, mock_get_tokenizer):
        mock_get_tokenizer.return_value = MagicMock()
        formatter = get_sft_formatter()
        assert formatter.packing is False
        assert formatter.response_only_loss is True


class TestTruncationEnforced:
    def test_long_example_truncated_to_max_length(self):
        with patch("baligh.data.formatter.get_tokenizer", return_value=FakeTokenizer()):
            formatter = PromptFormatter(max_seq_length=32)

        long_instruction = "كلمة " * 200
        result = formatter.format_sft(
            {"instruction": long_instruction, "input": "", "output": "جواب قصير"}
        )
        # Either the example survives truncated to max length...
        if result["input_ids"]:
            assert len(result["input_ids"]) <= 32
        else:
            # ...or nothing trainable survived truncation (skipped).
            assert result["input_ids"] == []
