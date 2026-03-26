from pathlib import Path
from transformers import DebertaV2Config
from transformers.models.auto import (
    AutoModelForSequenceClassification as SequenceClassification,
    AutoTokenizer as Tokenizer,
)
import sys

service_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(service_root)

from src import clean_dir

# Tested models for zero-shot learning
MODEL_LIST = [
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    "cross-encoder/nli-deberta-v3-large",
    "facebook/bart-large-mnli",
    "valhalla/distilbart-mnli-12-3",
    "typeform/distilbert-base-uncased-mnli",
]

THISMODEL_ = MODEL_LIST[0]

MYCACHEDIR = service_root + "/models/cache/"
PRETRAINEDMODELDIR = service_root + "/models/pretrained/nli_model"
HFCACHE = Path("~/.cache/huggingface/transformers/").expanduser().resolve()


if __name__ == "__main__":
    print("Pre Clean Cache ...")
    clean_dir(MYCACHEDIR)
    clean_dir(HFCACHE)
    clean_dir(PRETRAINEDMODELDIR)

    nli_model = SequenceClassification.from_pretrained(
        THISMODEL_, cache_dir=MYCACHEDIR, local_files_only=False
    )
    nli_model.save_pretrained(PRETRAINEDMODELDIR)

    USE_FAST = not isinstance(nli_model.config, DebertaV2Config)
    tokenizer = Tokenizer.from_pretrained(
        THISMODEL_, cache_dir=MYCACHEDIR, local_files_only=False, use_fast=USE_FAST
    )
    tokenizer.save_pretrained(PRETRAINEDMODELDIR)

    print("Post Clean Cache ...")
    clean_dir(MYCACHEDIR)
    clean_dir(HFCACHE)