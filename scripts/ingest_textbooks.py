import sys
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
        
    textbooks_dir = BASE_DIR / "data" / "textbooks"
    if not textbooks_dir.exists():
        print(f"❌ Textbooks directory not found at: {textbooks_dir}")
        sys.exit(1)
        
    # Supported suffixes
    allowed_suffixes = {".txt", ".md", ".text"}
    
    indexed_count = 0
    failed_count = 0
    
    # Traverse directory
    print(f"Scanning directory: {textbooks_dir} for files to index...")
    for path in textbooks_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            print(f"Found file: {path.relative_to(BASE_DIR)}")
            
            # Index file (this will automatically parse standard, subject, chapter)
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

if __name__ == '__main__':
    main()
