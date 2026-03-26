import transformers
from pathlib import Path

from transformers.models.auto import (
    AutoModelForSequenceClassification as SequenceClassification,
    AutoTokenizer as Tokenizer,
)
from .zero_shot_classification import ZeroShotClassification


TASK = "zero-shot-classification"
FRAMEWORK = "pt"

service_root = str(Path(__file__).resolve().parent.parent)
PRETRAINEDMODELDIR = service_root + "/models/pretrained/nli_model/"

# Load NLI model and tokenizer from the local pretrained directory
nli_model = SequenceClassification.from_pretrained(PRETRAINEDMODELDIR, local_files_only=True)
_use_fast = not isinstance(nli_model.config, transformers.DebertaV2Config)
tokenizer = Tokenizer.from_pretrained(PRETRAINEDMODELDIR, local_files_only=True, use_fast=_use_fast)

zero_shot_classifier = ZeroShotClassification(
    task=TASK, model=nli_model, tokenizer=tokenizer, framework=FRAMEWORK
)

