# Customer Review Intent Classification — Zero-Shot Learning

A zero-shot NLI-based classifier that categorizes customer reviews by intent — without any labeled training data.

## Overview

This project uses a pretrained Natural Language Inference (NLI) model ([DeBERTa-v3-large](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)) to classify customer reviews into intent categories via zero-shot learning. No fine-tuning or labeled datasets are required.

**Labels:** `complaint` · `inquiry` · `feedback` · `suggestion` · `request`

**Hypothesis template:** `"The customer intent in this review is {}"`

**Score range:** [0, 1] — higher means stronger match

## Project Structure

```
├── eds_entry.py                  # Main classification entry point
├── src/
│   ├── auto_zero_shot_classifier.py   # Loads pretrained model & tokenizer
│   ├── zero_shot_classification.py    # NLI zero-shot classification pipeline
│   ├── text_labels.py                 # Classification labels & threshold
│   ├── text_input_data.py             # Input data reading & processing
│   ├── text_preprocessing.py          # Text normalization & cleaning
│   └── utils.py                       # Utility functions
├── models/
│   ├── model_download_and_cache.py    # Download & cache the NLI model
│   └── pretrained/nli_model/          # Local model files (gitignored)
├── input_data/
│   └── input.tsv                      # Sample customer reviews (25 rows)
├── demo/
│   └── demo_textclassification.ipynb  # Jupyter notebook demo
├── sandbox/
│   └── test_tsv_file.py               # Batch classification script
├── streamlitApp/
│   └── myApp_singleEntry.py           # Streamlit web UI
├── result_dataframe/                  # Classification output (TSV files)
└── readme/
    └── environment-zeroshotclassification.yaml  # Conda environment
```

## Getting Started

### 1. Create the environment

```bash
conda env create -f readme/environment-zeroshotclassification.yaml
conda activate env-zeroshotclassification
```

### 2. Download the model

```bash
python models/model_download_and_cache.py
```

This downloads the DeBERTa-v3-large NLI model and saves it to `models/pretrained/nli_model/`.

### 3. Run classification

**Batch mode** (from the `sandbox/` directory):
```bash
cd sandbox
python test_tsv_file.py
```

**Streamlit UI:**
```bash
streamlit run streamlitApp/myApp_singleEntry.py
```

**Command line:**
```bash
python eds_entry.py -s "The product broke after two days" -l "complaint,inquiry,feedback,suggestion,request" -v
```

**Jupyter notebook:**
Open `demo/demo_textclassification.ipynb` and run all cells.

### 4. Input format

The input TSV file must have these columns (tab-separated):

| Column | Description |
|--------|-------------|
| `row`  | Row number |
| `id`   | Review ID |
| `title`| Short title/subject |
| `body` | Full review text |

## Model

- **Model:** [MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
- **Approach:** Natural Language Inference (NLI) — the model evaluates whether the review text *entails* each hypothesis (e.g., "The customer intent in this review is complaint")
- **Framework:** PyTorch + HuggingFace Transformers

## Dependencies

- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- Polars
- Pandas
- Streamlit
- contractions
- sentencepiece

## License

This project is for academic purposes.

