import sys
import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure modules directory is also on PATH
if str(BASE_DIR / "modules") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "modules"))

from modules.ai_query import get_rag_pipeline

def main():
    print("🚀 Starting Textbook Ingestion Script...")
    pipeline = get_rag_pipeline()
    if not pipeline:
        print("❌ Failed to initialize RAG Pipeline. Make sure settings.json is configured.")
        sys.exit(1)
        
    from modules.config_loader import data_directory
    parser = argparse.ArgumentParser(description="Index textbook text or reingest archived sources.")
    parser.add_argument("directory", nargs="?", type=Path, default=data_directory() / "textbooks")
    parser.add_argument("--sources", action="store_true", help="read saved source JSON from an earlier index")
    args = parser.parse_args()
    textbooks_dir = args.directory.expanduser().resolve()
    if not textbooks_dir.exists():
        print(f"❌ Textbooks directory not found at: {textbooks_dir}")
        sys.exit(1)
        
    # Supported suffixes
    allowed_suffixes = {".json"} if args.sources else {".txt", ".md", ".text"}
    
    indexed_count = 0
    failed_count = 0
    
    # Traverse directory
    print(f"Scanning directory: {textbooks_dir} for files to index...")
    for path in textbooks_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            print(f"Found file: {path.name}")
            
            # Index file (this will automatically parse standard, subject, chapter)
            if args.sources:
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    success = pipeline.index_document(record["text"], tags=record.get("tags", {}))
                except (ValueError, KeyError, OSError):
                    success = False
            else:
                success = pipeline.index_file(str(path))
            if success:
                print(f"  ✅ Successfully indexed!")
                indexed_count += 1
            else:
                print(f"  ❌ Failed to index.")
                failed_count += 1
                
    print("\n🏁 Ingestion Summary:")
    print(f"  Indexed: {indexed_count} files")
    print(f"  Failed:  {failed_count} files")
    return 1 if failed_count or not indexed_count else 0

if __name__ == '__main__':
    sys.exit(main())
