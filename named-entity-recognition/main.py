"""CLI entry point for PDF -> NER -> CSV flow."""

import argparse
import os
from datetime import datetime
from ner_pipeline import NERModelFactory
from pdf_extractor import PDFTextExtractor
from csv_exporter import CSVExporter


def split_text_chunks(text: str, max_chunk_size: int = 1000, overlap: int = 120) -> list[tuple[str, int]]:
    """Split text into overlapping chunks with source offsets."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start + (max_chunk_size // 2):
                end = split_at
        raw_chunk = text[start:end]
        chunk = raw_chunk.strip()
        if chunk:
            leading_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            chunk_start = start + leading_trim
            chunks.append((chunk, chunk_start))
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        start = next_start if next_start > start else end
    return chunks


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Remove duplicates introduced by overlapping chunks."""
    sorted_entities = sorted(
        entities,
        key=lambda x: (
            int(x.get('start', 0)),
            int(x.get('end', 0)),
            str(x.get('entity', '')),
            str(x.get('word', ''))
        )
    )

    deduped = []
    seen_exact = set()
    seen_near = {}

    for entity in sorted_entities:
        word = str(entity.get('word', '')).strip()
        label = str(entity.get('entity', '')).strip()
        start = int(entity.get('start', 0))
        end = int(entity.get('end', 0))
        if not word or not label:
            continue

        exact_key = (label, word.lower(), start, end)
        if exact_key in seen_exact:
            continue

        near_key = (label, word.lower())
        prev_pos = seen_near.get(near_key)
        if prev_pos and abs(start - prev_pos[0]) <= 3 and abs(end - prev_pos[1]) <= 3:
            continue

        seen_exact.add(exact_key)
        seen_near[near_key] = (start, end)
        deduped.append(entity)

    filtered = []
    for entity in deduped:
        word = str(entity.get('word', '')).strip().lower()
        label = str(entity.get('entity', '')).strip()
        start = int(entity.get('start', 0))
        end = int(entity.get('end', 0))

        is_contained = False
        for kept in filtered:
            kept_word = str(kept.get('word', '')).strip().lower()
            kept_label = str(kept.get('entity', '')).strip()
            kept_start = int(kept.get('start', 0))
            kept_end = int(kept.get('end', 0))

            same_label = label == kept_label
            span_contained = start >= kept_start and end <= kept_end
            text_overlaps = word in kept_word or kept_word in word
            if same_label and span_contained and text_overlaps:
                is_contained = True
                break

        if not is_contained:
            filtered.append(entity)

    return filtered


def process_pdf_with_ner(pdf_path: str, domain: str = 'general', 
                         output_dir: str = 'output') -> dict:
    """Run end-to-end processing for one PDF."""
    print("=" * 60)
    print("Named Entity Recognition Pipeline")
    print("=" * 60)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[1/4] Extracting text from PDF: {pdf_path}")
    extractor = PDFTextExtractor()
    
    try:
        pdf_info = extractor.get_pdf_info(pdf_path)
        print(f"  - Pages: {pdf_info['num_pages']}")
        
        text = extractor.extract_text(pdf_path)
        print(f"  - Extracted {len(text)} characters")
        
        if not text or len(text.strip()) < 10:
            raise ValueError("Extracted text is too short or empty")
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        raise
    
    print(f"\n[2/4] Loading {domain.upper()} NER model...")
    try:
        ner_pipeline = NERModelFactory.create(domain)
        print(f"  - Model loaded successfully")
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        raise
    
    print(f"\n[3/4] Running NER on extracted text...")
    try:
        # Chunk long documents to stay within model limits.
        max_chunk_size = 1000
        if len(text) > max_chunk_size:
            print(f"  - Text is long, processing in chunks...")
            chunks = split_text_chunks(text, max_chunk_size=max_chunk_size, overlap=120)
            all_entities = []
            for i, (chunk, chunk_start) in enumerate(chunks, 1):
                print(f"  - Processing chunk {i}/{len(chunks)}")
                entities = ner_pipeline.predict(chunk)
                for entity in entities:
                    adjusted = dict(entity)
                    if isinstance(adjusted.get('start'), int):
                        adjusted['start'] += chunk_start
                    if isinstance(adjusted.get('end'), int):
                        adjusted['end'] += chunk_start
                    all_entities.append(adjusted)
        else:
            all_entities = ner_pipeline.predict(text)

        all_entities = deduplicate_entities(all_entities)
        
        print(f"  - Found {len(all_entities)} entities")
        
        entity_counts = {}
        for entity in all_entities:
            entity_type = entity.get('entity', 'UNKNOWN')
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        print("  - Entity breakdown:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"    * {entity_type}: {count}")
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        raise
    
    print(f"\n[4/4] Exporting results to CSV...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    output_file = os.path.join(output_dir, f"ner_results_{pdf_name}_{timestamp}.csv")
    summary_file = os.path.join(output_dir, f"ner_summary_{pdf_name}_{timestamp}.csv")
    
    try:
        exporter = CSVExporter()
        
        exporter.export_entities(all_entities, output_file, include_metadata=True)
        print(f"  - Detailed results: {output_file}")
        
        exporter.export_summary_statistics(all_entities, summary_file)
        print(f"  - Summary statistics: {summary_file}")
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        raise
    
    print("\n" + "=" * 60)
    print("Processing completed successfully!")
    print("=" * 60)
    
    return {
        'entities': all_entities,
        'entity_counts': entity_counts,
        'output_file': output_file,
        'summary_file': summary_file,
        'total_entities': len(all_entities)
    }


def main():
    """Command-line interface for the NER pipeline."""
    parser = argparse.ArgumentParser(
        description='Extract named entities from PDF documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.pdf
  python main.py input.pdf --domain medical
  python main.py input.pdf --domain financial --output results/
  
Available domains:
  - general: Person, Location, Organization
  - medical: Medical entities and diseases
  - financial: Financial entities
        """
    )
    
    parser.add_argument('pdf_path', help='Path to the input PDF file')
    parser.add_argument('--domain', '-d', default='general',
                       choices=['general', 'medical', 'financial'],
                       help='NER domain/model to use (default: general)')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory for results (default: output/)')
    
    args = parser.parse_args()
    
    try:
        results = process_pdf_with_ner(
            pdf_path=args.pdf_path,
            domain=args.domain,
            output_dir=args.output
        )
        
        print(f"\nSummary:")
        print(f"  Total entities found: {results['total_entities']}")
        print(f"  Results saved to: {results['output_file']}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
