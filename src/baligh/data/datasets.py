"""Dataset registry for Baligh-1.5B v0."""

from baligh.utils.logging import get_logger

logger = get_logger(__name__)

DATASETS = {
    'arabicweb24': {
        'path': 'lightonai/ArabicWeb24', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 28000000000, 'domain': 'web', 'type': 'cpt',
    },
    'arabictext_large': {
        'path': 'Jr23xd23/ArabicText-Large', 'split': 'train', 'streaming': False, 'text_column': 'text', 'license': 'Apache-2.0', 'tokens': 1000000000, 'domain': 'general', 'type': 'cpt',
    },
    'arabic_pile': {
        'path': 'premio-ai/TheArabicPile_Dialects', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 5000000000, 'domain': 'mixed', 'type': 'cpt',
    },
    'oscar_ar': {
        'path': 'oscar-corpus/oscar_ar', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 10000000000, 'domain': 'web', 'type': 'cpt',
    },
    'mc4_ar': {
        'path': 'google/mc4', 'name': 'ar', 'split': 'train', 'streaming': True, 'text_column': 'text', 'license': 'Open', 'tokens': 50000000000, 'domain': 'web', 'type': 'cpt',
    },
    'hadith_datasets': {
        'path': 'meeAtif/hadith_datasets', 'split': 'train', 'streaming': False, 'text_column': 'text', 'license': 'Open', 'domain': 'islamic', 'type': 'cpt_islamic',
    },
    'quran_qa': {
        'path': 'nazimali/quran-question-answer-context', 'split': 'train', 'streaming': False, 'text_column': 'context', 'license': 'CC', 'domain': 'islamic', 'type': 'cpt_islamic',
    },
    'quran_md': {
        'path': 'Buraaq/quran-audio-text-dataset', 'split': 'train', 'streaming': False, 'text_column': 'text', 'license': 'Open', 'domain': 'islamic', 'type': 'cpt_islamic',
    },
    'cidar': {
        'path': 'arbml/CIDAR', 'split': 'train', 'streaming': False, 'instruction_column': 'instruction', 'input_column': 'input', 'output_column': 'output', 'license': 'Apache-2.0', 'domain': 'instruction', 'type': 'sft',
    },
    'evol_instruct_arabic': {
        'path': 'FreedomIntelligence/evol-instruct-arabic', 'split': 'train', 'streaming': False, 'instruction_column': 'instruction', 'input_column': 'input', 'output_column': 'output', 'license': 'Open', 'domain': 'instruction', 'type': 'sft',
    },
    'gazelle': {
        'path': 'Gazelle/arabic-writing', 'split': 'train', 'streaming': False, 'instruction_column': 'instruction', 'input_column': 'input', 'output_column': 'output', 'license': 'Open', 'domain': 'instruction', 'type': 'sft',
    },
    'summarization': {
        'path': 'BounharAbdelaziz/arabic-msa-summarization', 'split': 'train', 'streaming': False, 'instruction_column': 'text', 'output_column': 'summary', 'license': 'Open', 'domain': 'summarization', 'type': 'sft',
    },
    'mmlu_arabic': {
        'path': 'FreedomIntelligence/MMLU_Arabic', 'split': 'test', 'streaming': False, 'license': 'Open', 'domain': 'eval', 'type': 'eval',
    },
    'cidar_eval': {
        'path': 'arbml/CIDAR', 'split': 'test', 'streaming': False, 'license': 'Apache-2.0', 'domain': 'eval', 'type': 'eval',
    },
    'mr_tydi_arabic': {
        'path': 'castorini/mr-tydi', 'name': 'arabic', 'split': 'test', 'streaming': False, 'license': 'Open', 'domain': 'eval', 'type': 'eval',
    },
}

def get_dataset_config(name):
    if name not in DATASETS:
        raise ValueError('Unknown dataset: %s' % name)
    return DATASETS[name]

def list_datasets(dataset_type=None):
    if dataset_type:
        return {k: v for k, v in DATASETS.items() if v.get('type') == dataset_type}
    return DATASETS

def get_cpt_datasets():
    return list_datasets('cpt')

def get_sft_datasets():
    return list_datasets('sft')

def get_eval_datasets():
    return list_datasets('eval')

def get_islamic_datasets():
    return list_datasets('cpt_islamic')
