"""Helpers for writing NER outputs to CSV files."""

import pandas as pd
from typing import List, Dict
from datetime import datetime
import os


class CSVExporter:
    """Write entity-level and summary CSV files."""
    
    @staticmethod
    def export_entities(entities: List[Dict], output_path: str, 
                       include_metadata: bool = True) -> str:
        """Export entities to a CSV file."""
        if not entities:
            # Keep expected schema even when there are no entities.
            df = pd.DataFrame(columns=['entity_text', 'entity_type', 'confidence'])
        else:
            df_data = []
            for entity in entities:
                row = {
                    'entity_text': entity.get('word', ''),
                    'entity_type': entity.get('entity', ''),
                }
                
                if include_metadata:
                    row['start_position'] = entity.get('start', '')
                    row['end_position'] = entity.get('end', '')
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
        
        if include_metadata and len(df) > 0:
            df['extraction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
    
    @staticmethod
    def export_entities_with_context(entities: List[Dict], original_text: str,
                                     output_path: str, context_window: int = 50) -> str:
        """Export entities with a lightweight context placeholder."""
        if not entities:
            df = pd.DataFrame(columns=['entity_text', 'entity_type', 'context'])
        else:
            df_data = []
            
            for entity in entities:
                # Placeholder context for now; can be upgraded with true spans later.
                row = {
                    'entity_text': entity.get('word', ''),
                    'entity_type': entity.get('entity', ''),
                    'context': f"...{entity.get('word', '')}..."
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        return output_path
    
    @staticmethod
    def export_summary_statistics(entities: List[Dict], output_path: str) -> str:
        """Export counts and percentages by entity type."""
        if not entities:
            df = pd.DataFrame(columns=['entity_type', 'count', 'percentage'])
        else:
            entity_counts = {}
            for entity in entities:
                entity_type = entity.get('entity', 'UNKNOWN')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            total = len(entities)
            summary_data = []
            for entity_type, count in sorted(entity_counts.items()):
                summary_data.append({
                    'entity_type': entity_type,
                    'count': count,
                    'percentage': f"{(count/total)*100:.2f}%"
                })
            
            df = pd.DataFrame(summary_data)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        return output_path


if __name__ == "__main__":
    sample_entities = [
        {'word': 'Apple Inc.', 'entity': 'ORG', 'start': 0, 'end': 2},
        {'word': 'Cupertino', 'entity': 'LOC', 'start': 5, 'end': 5},
        {'word': 'California', 'entity': 'LOC', 'start': 7, 'end': 7},
        {'word': 'Tim Cook', 'entity': 'PER', 'start': 9, 'end': 10},
    ]
    
    exporter = CSVExporter()
    output_file = "output/sample_output.csv"
    
    os.makedirs("output", exist_ok=True)
    exporter.export_entities(sample_entities, output_file)
    
    print(f"Exported entities to: {output_file}")
    
    summary_file = "output/sample_summary.csv"
    exporter.export_summary_statistics(sample_entities, summary_file)
    print(f"Exported summary to: {summary_file}")
