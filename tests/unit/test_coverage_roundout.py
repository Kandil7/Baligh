"""Coverage round-out: quantization guards, logging, distributed, datasets."""

from unittest.mock import MagicMock, patch

import pytest


class TestQuantizeGuards:
    def test_gguf_fails_fast_without_tooling(self):
        """v0.1 called a nonexistent module; now it must fail loudly."""
        from baligh.models.quantization import quantize_gguf

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="llama.cpp"):
                quantize_gguf("model_dir", "out.gguf")

    def test_gptq_requires_calibration(self):
        """GPTQ without calibration data produces an invalid artifact — refuse."""
        from baligh.models.quantization import quantize_gptq

        with pytest.raises(ValueError, match="calibration"):
            quantize_gptq("model_dir", "out_dir")


class TestSetupLogging:
    def test_text_mode_smoke(self, tmp_path):
        from baligh.utils.logging import setup_logging

        log_file = tmp_path / "logs" / "test.log"
        setup_logging(log_level="INFO", log_file=log_file)
        assert log_file.parent.exists()

    def test_json_console_mode(self):
        from baligh.utils.logging import setup_logging

        setup_logging(log_level="INFO", log_format="json")


class TestDistributedSingleProcess:
    def test_single_process_paths(self):
        from baligh.utils import distributed as dist_utils

        assert dist_utils.is_distributed() in (True, False)
        if not dist_utils.is_distributed():
            assert dist_utils.get_world_size() == 1
            assert dist_utils.get_rank() == 0
            assert dist_utils.is_main_process() is True

    def test_reduce_dict_passthrough_when_not_distributed(self):
        from baligh.utils.distributed import reduce_dict

        payload = {"loss": MagicMock()}
        assert reduce_dict(payload) is payload

    def test_gather_object_passthrough(self):
        from baligh.utils.distributed import gather_object

        assert gather_object({"a": 1}) == [{"a": 1}]


class TestTypedDatasetLoaders:
    @patch("baligh.data.loader.load_dataset_by_name")
    def test_load_cpt_datasets_filters_by_type(self, mock_load):
        from baligh.data.datasets import load_cpt_datasets

        mock_load.return_value = MagicMock()
        result = load_cpt_datasets()
        assert set(result.keys()) == {
            "arabicweb24",
            "arabictext_large",
            "arabic_pile",
            "oscar_ar",
            "mc4_ar",
        }

    @patch("baligh.data.loader.load_dataset_by_name")
    def test_load_sft_datasets_returns_all_four(self, mock_load):
        from baligh.data.datasets import DATASETS, load_sft_datasets

        mock_load.return_value = MagicMock()
        expected = {k for k, v in DATASETS.items() if v.get("type") == "sft"}
        assert set(load_sft_datasets().keys()) == expected

    def test_getters_return_registry_slices(self):
        from baligh.data.datasets import (
            DATASETS,
            get_cpt_datasets,
            get_eval_datasets,
            get_islamic_datasets,
            get_sft_datasets,
        )

        assert all(v.get("type") == "cpt" for v in get_cpt_datasets().values())
        assert all(v.get("type") == "sft" for v in get_sft_datasets().values())
        assert all(v.get("type") == "eval" for v in get_eval_datasets().values())
        assert all(v.get("type") == "cpt_islamic" for v in get_islamic_datasets().values())
        assert len(DATASETS) >= 14


class TestMemoryEstimatorQLoRA:
    def test_trainable_fraction_shrinks_grads_and_optimizer(self):
        from baligh.utils.memory import estimate_training_memory

        full = estimate_training_memory(1_500_000_000, 2, 2048)
        qlora = estimate_training_memory(1_500_000_000, 2, 2048, trainable_fraction=0.01)
        assert qlora["gradients_gb"] < full["gradients_gb"] / 10
        assert qlora["optimizer_gb"] < full["optimizer_gb"] / 10

    def test_tracker_on_cpu_does_not_crash(self):
        from baligh.utils.memory import MemoryTracker

        tracker = MemoryTracker()
        tracker.snapshot("t")
        summary = tracker.summary()
        assert isinstance(summary, dict)


class TestSampledValidation:
    def test_max_samples_bounds_iteration(self):
        from baligh.data.validators import validate_dataset

        class CountingIterable:
            def __init__(self, n):
                self.n = n
                self.iterated = 0

            def __iter__(self):
                self.iterated += 1
                return iter(
                    {"text": "نص تجريبي طويل بما يكفي لتجاوز الحد الأدنى للتحقق"}
                    for _ in range(self.n)
                )

        ds = CountingIterable(100000)
        valid, invalid, errors = validate_dataset(ds, dataset_type="cpt", max_samples=5)
        assert valid + invalid == 5


class TestCleaningSanitize:
    def test_sanitize_preserves_row_alignment(self):
        """sanitize returns '' instead of dropping the row."""
        from baligh.data.cleaner import CleaningPipeline

        pipeline = CleaningPipeline(language_filter="arabic")
        kept = pipeline.sanitize("هذا نص عربي واضح ومفهوم للجميع بكل تأكيد")
        blanked = pipeline.sanitize("totally english text here")
        assert kept and isinstance(blanked, str) and blanked == ""


class TestSeedingDeterminismEnv:
    def test_cublas_workspace_config_set(self, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        import torch

        from baligh.utils.seeding import set_seed

        if torch.cuda.is_available():
            set_seed(7)
            import os

            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
