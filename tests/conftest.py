"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_cpt_text():
    """Sample CPT text for testing."""
    return "هذا نص تجريبي عربي طويل يحتوي على الكثير من الكلمات والمعلومات المتنوعة في اللغة العربية"


@pytest.fixture
def sample_sft_example():
    """Sample SFT example for testing."""
    return {
        "instruction": "اكتب قصيدة قصيرة عن القدس",
        "input": "",
        "output": "القدس مدينة القداسة\nفيها الأقصى والمقدسات\nتحتضن التاريخ والأصالة"
    }


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {"text": "نص عربي أول طويل يحتوي على كلمات متنوعة ومختلفة"},
        {"text": "نص عربي ثاني طويل أيضاً يحتوي على معلومات مفيدة"},
        {"text": "نص عربي ثالث طويل يحتوي على محتوى عربي أصيل"},
    ]
