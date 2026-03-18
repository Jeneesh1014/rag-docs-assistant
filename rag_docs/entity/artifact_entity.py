
# Artifact Entities = What results come OUT of each step

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestionArtifact:
   
    chroma_db_path: Path

    collection_name: str

    total_pages_loaded: int

    total_chunks_stored: int

    total_files_processed: int

    verification_passed: bool

    log_file_path: Path