"""Tests for dataset mixing."""

from unittest.mock import MagicMock, patch

import pytest

from baligh.data.mixer import (
    DatasetMixer,
    get_cpt_mixer,
    get_sft_mixer,
    mix_cpt_datasets,
    mix_sft_datasets,
)


class TestDatasetMixer:
    """Tests for DatasetMixer."""

    def test_init(self):
        mixer = DatasetMixer({"a": 0.7, "b": 0.3})
        assert mixer.mix_config == {"a": 0.7, "b": 0.3}
        assert mixer.seed == 42
        assert mixer.stopping_strategy == "first_exhausted"

    def test_mix_basic(self):
        mock_dataset_a = MagicMock()
        mock_dataset_b = MagicMock()
        datasets = {"a": mock_dataset_a, "b": mock_dataset_b}

        with patch("baligh.data.mixer.interleave_datasets") as mock_interleave:
            mock_interleave.return_value = MagicMock()

            mixer = DatasetMixer({"a": 0.7, "b": 0.3})
            result = mixer.mix(datasets)

            mock_interleave.assert_called_once()
            call_args = mock_interleave.call_args
            assert call_args[1]["probabilities"] == [0.7, 0.3]

    def test_mix_normalizes_ratios(self):
        mock_dataset_a = MagicMock()
        mock_dataset_b = MagicMock()
        datasets = {"a": mock_dataset_a, "b": mock_dataset_b}

        with patch("baligh.data.mixer.interleave_datasets") as mock_interleave:
            mock_interleave.return_value = MagicMock()

            mixer = DatasetMixer({"a": 70, "b": 30})
            result = mixer.mix(datasets)

            call_args = mock_interleave.call_args
            assert call_args[1]["probabilities"] == [0.7, 0.3]

    def test_mix_skips_missing_dataset(self):
        mock_dataset_a = MagicMock()
        datasets = {"a": mock_dataset_a}

        with patch("baligh.data.mixer.interleave_datasets") as mock_interleave:
            mock_interleave.return_value = MagicMock()

            mixer = DatasetMixer({"a": 0.7, "b": 0.3})
            result = mixer.mix(datasets)

            call_args = mock_interleave.call_args
            assert len(call_args[0][0]) == 1

    def test_mix_raises_on_no_datasets(self):
        mixer = DatasetMixer({"a": 0.7, "b": 0.3})

        with pytest.raises(ValueError) as excinfo:
            mixer.mix({})
        assert "No valid datasets to mix" in str(excinfo.value)

    def test_strict_mode_raises_on_unknown_key(self):
        """strict=True: declared ratios can never silently drift."""
        mixer = DatasetMixer({"a": 0.7, "phantom": 0.3}, strict=True)
        with pytest.raises(ValueError) as excinfo:
            mixer.mix({"a": MagicMock()})
        assert "phantom" in str(excinfo.value)

    def test_default_mode_skips_unknown_key(self):
        mixer = DatasetMixer({"a": 0.7, "phantom": 0.3}, strict=False)
        with patch("baligh.data.mixer.interleave_datasets") as mock_interleave:
            mock_interleave.return_value = MagicMock()
            mixer.mix({"a": MagicMock()})
            call_args = mock_interleave.call_args
            assert len(call_args[0][0]) == 1


class TestFactoryFunctions:
    """Tests for mixer factory functions."""

    @patch("baligh.data.mixer.get_cpt_config")
    def test_get_cpt_mixer(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.dataset_mix = {"a": 0.7, "b": 0.3}
        mock_get_config.return_value = mock_config

        mixer = get_cpt_mixer()
        assert mixer.mix_config == {"a": 0.7, "b": 0.3}

    @patch("baligh.data.mixer.get_sft_config")
    def test_get_sft_mixer(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.dataset_mix = {"x": 0.5, "y": 0.5}
        mock_get_config.return_value = mock_config

        mixer = get_sft_mixer()
        assert mixer.mix_config == {"x": 0.5, "y": 0.5}

    @patch("baligh.data.mixer.get_cpt_mixer")
    def test_mix_cpt_datasets(self, mock_get_mixer):
        mock_mixer = MagicMock()
        mock_mixer.mix.return_value = MagicMock()
        mock_get_mixer.return_value = mock_mixer

        datasets = {"a": MagicMock()}
        result = mix_cpt_datasets(datasets, seed=42)

        mock_mixer.mix.assert_called_once_with(datasets)

    @patch("baligh.data.mixer.get_sft_mixer")
    def test_mix_sft_datasets(self, mock_get_mixer):
        mock_mixer = MagicMock()
        mock_mixer.mix.return_value = MagicMock()
        mock_get_mixer.return_value = mock_mixer

        datasets = {"x": MagicMock()}
        result = mix_sft_datasets(datasets, seed=42)

        mock_mixer.mix.assert_called_once_with(datasets)
