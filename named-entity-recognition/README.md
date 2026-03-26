# Named Entity Recognition (NER) Assignment

This project extracts named entities from PDF files and exports the results to CSV.
It includes:

- a custom NER pipeline (no direct use of `transformers.pipeline(...)`)
- a command-line workflow
- a Streamlit app for interactive use (bonus requirement)

## Assignment Requirement Mapping

- Custom NER implementation without Hugging Face pipeline API: yes (`ner_pipeline.py`)
- At least one domain model (medical, financial, or general): yes (three domains supported)
- PDF text extraction using PyMuPDF or similar: yes (`pdf_extractor.py`)
- CSV output with entity labels: yes (`csv_exporter.py`)
- Bonus virtual environment config: yes (`requirements.txt` and `environment.yml`)
- Bonus Streamlit upload/process/download flow: yes (`app.py`)

## Project Structure

```text
ner_assignment/
|-- app.py
|-- main.py
|-- ner_pipeline.py
|-- pdf_extractor.py
|-- csv_exporter.py
|-- generate_sample_pdf.py
|-- test_pipeline.py
|-- requirements.txt
|-- environment.yml
|-- data/
`-- output/
```

## Setup

### Option A: pip + venv (recommended)

Windows PowerShell:

```powershell
cd c:\Users\shiva\ner_assignment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate ner_assignment
```

## Quick Start

### 1) Generate sample PDF

```bash
python generate_sample_pdf.py
```

This creates `data/sample_document.pdf`.

### 2) Run CLI pipeline

```bash
python main.py data/sample_document.pdf --domain general
```

Other domains:

```bash
python main.py data/sample_document.pdf --domain medical
python main.py data/sample_document.pdf --domain financial
```

### 3) Run Streamlit app (bonus)

```bash
streamlit run app.py
```

In the app:

1. Upload a PDF
2. Select domain
3. Click `Run NER Analysis`
4. Review table/chart
5. Download CSV

## Output

CLI mode writes files to `output/`:

- `ner_results_<pdf>_<timestamp>.csv`
- `ner_summary_<pdf>_<timestamp>.csv`

Detailed result columns:

- `entity_text`
- `entity_type`
- `start_position`
- `end_position`
- `extraction_timestamp`

Summary result columns:

- `entity_type`
- `count`
- `percentage`

## Technical Notes

- The project uses `AutoTokenizer` + `AutoModelForTokenClassification` directly.
- It does not call `transformers.pipeline(...)`.
- Long documents are chunked with overlap.
- Output quality is improved with:
  - label-aware confidence thresholds
  - duplicate removal across chunk overlap
  - basic noise filtering for common PDF artifacts

## Supported Models

- `general`: `dslim/bert-base-NER`
- `medical`: `alvaroalon2/biobert_diseases_ner`
- `financial`: `dbmdz/bert-large-cased-finetuned-conll03-english`

## Basic Validation

```bash
python test_pipeline.py
python generate_sample_pdf.py
python main.py data/sample_document.pdf --domain general
```

## Known Limitations

- If a PDF is scanned (image-only), text extraction may be weak and NER quality will drop.
- Table-heavy PDFs can produce fragmented text, which may create noisy entities.
- First model run can be slower because weights are downloaded and cached.

## Submission Checklist

- Include source code files (`*.py`)
- Include `requirements.txt` and/or `environment.yml`
- Include at least one `ner_results_*.csv` and one `ner_summary_*.csv`
- Keep email subject exactly: `NLP-Homework-NER`

---

If you want to verify quickly as a grader: run `python main.py data/sample_document.pdf --domain general` and inspect the generated CSV in `output/`.
