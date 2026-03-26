"""Custom NER pipeline built on tokenizer + model APIs."""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple
import re


class CustomNERPipeline:
    """Runs token classification and assembles entity spans."""
    
    def __init__(self, model_name: str, device: str = None, min_confidence: float = 0.60):
        """Load model + tokenizer and set filtering thresholds."""
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_confidence = min_confidence
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        # Per-label thresholds help keep noisy PDF entities in check.
        self.label_confidence_thresholds = {
            'PER': 0.60,
            'ORG': 0.66,
            'LOC': 0.66,
            'MISC': 0.82
        }
        
    def _tokenize_and_align(self, text: str) -> Tuple[torch.Tensor, List[str], List[int], List[Tuple[int, int]]]:
        """Tokenize text and return token/offset alignment metadata."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        word_ids = encoding.word_ids(batch_index=0)
        offsets = encoding['offset_mapping'][0].tolist()
        
        return input_ids, tokens, word_ids, offsets
    
    @staticmethod
    def _clean_entity_text(value: str) -> str:
        """Normalize whitespace and punctuation spacing in extracted entities."""
        value = value.replace("\n", " ").replace("\t", " ")
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\s+([,.;:!?])", r"\1", value)
        return value

    @staticmethod
    def _expand_span_to_word_boundaries(text: str, start_char: int, end_char: int) -> Tuple[int, int]:
        """Expand token-based span to nearest alphabetic word boundaries."""
        start = max(0, start_char)
        end = min(len(text), end_char)

        while start > 0 and text[start - 1].isalpha():
            start -= 1
        while end < len(text) and text[end].isalpha():
            end += 1

        return start, end

    def _should_keep_entity(self, text: str, entity_type: str, score: float) -> bool:
        """Filter low-value noisy entities commonly produced from messy PDFs."""
        threshold = self.label_confidence_thresholds.get(entity_type, self.min_confidence)
        if score < threshold:
            return False
        if len(text) < 2:
            return False
        if not any(ch.isalpha() for ch in text):
            return False
        if len(text.split()) > 6:
            return False
        if len(text) <= 2 and text.isalpha():
            return False
        if text.isupper() and len(text) <= 4:
            return False
        if text.endswith(".") and len(text) <= 4:
            return False

        lowered = text.lower()
        noise_phrases = {
            "python", "sql", "nosql", "cloud", "data", "analytics", "analysis",
            "learning", "model", "models", "visualization", "resources", "tools",
            "certificate", "course", "courses"
        }
        if lowered in noise_phrases:
            return False

        noise_tokens = {
            "THE", "THIS", "THAT", "AND", "OR", "OF", "IN", "ON", "FOR", "WITH",
            "BY", "TO", "FROM", "IS", "ARE", "WAS", "WERE"
        }
        if text.upper() in noise_tokens:
            return False
        return True

    def _aggregate_entities(self, tokens: List[str], predictions: List[int],
                           confidences: List[float], word_ids: List[int],
                           offsets: List[Tuple[int, int]], text: str) -> List[Dict]:
        """Merge token-level labels into final entity spans."""
        entities = []
        current_entity = None

        def finalize_current():
            nonlocal current_entity
            if not current_entity:
                return
            span_start, span_end = self._expand_span_to_word_boundaries(
                text,
                current_entity['start_char'],
                current_entity['end_char']
            )
            raw = text[span_start:span_end]
            cleaned = self._clean_entity_text(raw)
            score = sum(current_entity['scores']) / len(current_entity['scores'])
            entity_type = current_entity['entity']
            if self._should_keep_entity(cleaned, entity_type, score):
                entities.append({
                    'word': cleaned,
                    'entity': entity_type,
                    'start': span_start,
                    'end': span_end,
                    'score': round(score, 4)
                })
            current_entity = None
        
        for token, pred_id, token_score, word_id, offset in zip(
            tokens, predictions, confidences, word_ids, offsets
        ):
            if word_id is None:
                continue

            label = self.id2label[pred_id]
            start_char, end_char = offset
            if end_char <= start_char:
                continue
            
            # "O" means no entity at this token.
            if label == 'O':
                finalize_current()
                continue
            
            # Split BIO tag into prefix + type.
            if '-' in label:
                prefix, entity_type = label.split('-', 1)
            else:
                prefix = 'B'
                entity_type = label
            
            can_continue = (
                current_entity is not None and
                prefix == 'I' and
                current_entity['entity'] == entity_type and
                start_char <= current_entity['end_char'] + 2
            )

            # Start new span unless this token continues current one.
            if not can_continue:
                finalize_current()
                current_entity = {
                    'entity': entity_type,
                    'start_char': start_char,
                    'end_char': end_char,
                    'scores': [float(token_score)]
                }
            else:
                current_entity['end_char'] = max(current_entity['end_char'], end_char)
                current_entity['scores'].append(float(token_score))
        
        finalize_current()
        
        return entities
    
    def predict(self, text: str) -> List[Dict]:
        """Run NER for one text block."""
        if not text or not text.strip():
            return []
        
        input_ids, tokens, word_ids, offsets = self._tokenize_and_align(text)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]
            predictions = torch.argmax(logits, dim=-1).cpu().tolist()
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
        
        entities = self._aggregate_entities(tokens, predictions, confidences, word_ids, offsets, text)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Run NER for multiple text blocks."""
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append(entities)
        return results


class NERModelFactory:
    """Creates pipeline instances for supported domains."""
    
    MODELS = {
        'medical': 'alvaroalon2/biobert_diseases_ner',
        'financial': 'dbmdz/bert-large-cased-finetuned-conll03-english',
        'general': 'dslim/bert-base-NER',
        'general_large': 'dbmdz/bert-large-cased-finetuned-conll03-english'
    }
    
    @classmethod
    def create(cls, domain: str = 'general', device: str = None) -> CustomNERPipeline:
        """Build a pipeline for the requested domain."""
        if domain not in cls.MODELS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(cls.MODELS.keys())}")
        
        model_name = cls.MODELS[domain]
        return CustomNERPipeline(model_name, device)
    
    @classmethod
    def list_domains(cls) -> List[str]:
        """Return supported domain names."""
        return list(cls.MODELS.keys())


if __name__ == "__main__":
    pipeline = NERModelFactory.create('general')
    
    sample_text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
    entities = pipeline.predict(sample_text)
    
    print("\nExtracted Entities:")
    for entity in entities:
        print(f"  {entity['word']} -> {entity['entity']}")
