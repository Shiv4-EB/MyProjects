"""Streamlit UI for running NER on uploaded PDFs."""

import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
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


def initialize_session_state():
    """Initialize Streamlit session state fields."""
    if 'entities' not in st.session_state:
        st.session_state.entities = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False


def process_pdf(pdf_file, domain: str):
    """Run extraction + NER for one uploaded PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        with st.spinner('Extracting text from PDF...'):
            extractor = PDFTextExtractor()
            text = extractor.extract_text(tmp_path)
            
            if not text or len(text.strip()) < 10:
                st.error("Could not extract sufficient text from the PDF.")
                return None, None
            
            st.success(f"Extracted {len(text)} characters from PDF")
        
        with st.spinner(f'Loading {domain.upper()} NER model...'):
            ner_pipeline = NERModelFactory.create(domain)
            st.success("Model loaded successfully")
        
        with st.spinner('Running Named Entity Recognition...'):
            max_chunk_size = 1000
            if len(text) > max_chunk_size:
                chunks = split_text_chunks(text, max_chunk_size=max_chunk_size, overlap=120)
                progress_bar = st.progress(0)
                all_entities = []
                
                for i, (chunk, chunk_start) in enumerate(chunks):
                    entities = ner_pipeline.predict(chunk)
                    for entity in entities:
                        adjusted = dict(entity)
                        if isinstance(adjusted.get('start'), int):
                            adjusted['start'] += chunk_start
                        if isinstance(adjusted.get('end'), int):
                            adjusted['end'] += chunk_start
                        all_entities.append(adjusted)
                    progress_bar.progress((i + 1) / len(chunks))
            else:
                all_entities = ner_pipeline.predict(text)

            all_entities = deduplicate_entities(all_entities)
            
            st.success(f"Found {len(all_entities)} entities")
        
        if all_entities:
            df_data = []
            for entity in all_entities:
                df_data.append({
                    'Entity Text': entity.get('word', ''),
                    'Entity Type': entity.get('entity', ''),
                    'Start Position': entity.get('start', ''),
                    'End Position': entity.get('end', '')
                })
            results_df = pd.DataFrame(df_data)
        else:
            results_df = pd.DataFrame(columns=['Entity Text', 'Entity Type'])
        
        return all_entities, results_df
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    """Render and run the Streamlit app."""
    st.set_page_config(
        page_title="NER PDF Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📄 Named Entity Recognition - PDF Analyzer")
    st.markdown("""
    Upload a PDF document to extract and identify named entities such as persons, 
    organizations, locations, and more using state-of-the-art NER models.
    """)
    
    st.sidebar.header("⚙️ Configuration")
    
    domain = st.sidebar.selectbox(
        "Select NER Domain",
        options=['general', 'medical', 'financial'],
        help="Choose the domain-specific model for entity recognition"
    )
    
    st.sidebar.markdown("""
    ### Domain Information
    - **General**: Recognizes persons, locations, organizations, and miscellaneous entities
    - **Medical**: Specialized for medical terms and diseases
    - **Financial**: Optimized for financial entities
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to analyze"
        )
    
    with col2:
        st.header("Quick Info")
        st.info(f"""
        **Selected Domain:** {domain.upper()}
        
        **Status:** {'Ready to process' if uploaded_file else 'Waiting for file...'}
        """)
    
    if uploaded_file is not None:
        st.markdown("---")
        
        if st.button("🚀 Run NER Analysis", type="primary", use_container_width=True):
            entities, results_df = process_pdf(uploaded_file, domain)
            
            if entities is not None:
                st.session_state.entities = entities
                st.session_state.results_df = results_df
                st.session_state.processing_complete = True
    
    if st.session_state.processing_complete and st.session_state.results_df is not None:
        st.markdown("---")
        st.header("📊 Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Entities Found",
                len(st.session_state.entities)
            )
        
        with col2:
            if len(st.session_state.entities) > 0:
                unique_types = st.session_state.results_df['Entity Type'].nunique()
                st.metric("Unique Entity Types", unique_types)
            else:
                st.metric("Unique Entity Types", 0)
        
        with col3:
            if len(st.session_state.entities) > 0:
                most_common = st.session_state.results_df['Entity Type'].mode()[0]
                st.metric("Most Common Type", most_common)
            else:
                st.metric("Most Common Type", "N/A")
        
        if len(st.session_state.entities) > 0:
            st.subheader("Entity Type Distribution")
            entity_counts = st.session_state.results_df['Entity Type'].value_counts()
            st.bar_chart(entity_counts)
        
        st.subheader("Extracted Entities")
        
        if len(st.session_state.entities) > 0:
            entity_types = ['All'] + sorted(st.session_state.results_df['Entity Type'].unique().tolist())
            selected_type = st.selectbox("Filter by Entity Type", entity_types)
            
            if selected_type != 'All':
                filtered_df = st.session_state.results_df[
                    st.session_state.results_df['Entity Type'] == selected_type
                ]
            else:
                filtered_df = st.session_state.results_df
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = st.session_state.results_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"ner_results_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                summary_df = st.session_state.results_df['Entity Type'].value_counts().reset_index()
                summary_df.columns = ['Entity Type', 'Count']
                summary_df['Percentage'] = (summary_df['Count'] / summary_df['Count'].sum() * 100).round(2).astype(str) + '%'
                summary_csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Summary (CSV)",
                    data=summary_csv,
                    file_name=f"ner_summary_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No entities found in the document.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>NLP Assignment - Named Entity Recognition Pipeline</p>
        <p>Built with Streamlit, Transformers, and PyMuPDF</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
