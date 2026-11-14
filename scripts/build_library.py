import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm.base import ProjectIdea


def validate_library():
    library_dir = os.path.join(os.path.dirname(__file__), "..", "data", "projects")
    if not os.path.exists(library_dir):
        print(f"[!] Library directory {library_dir} not found.")
        return False

    valid_count = 0
    errors = 0

    print("=" * 80)
    print("Curated Project Library Validator")
    print("=" * 80)

    for fname in sorted(os.listdir(library_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(library_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate with Pydantic
            project = ProjectIdea.model_validate(data)
            print(f"✅ {fname:<32} | {project.ProjectTitle[:40]:<40} | {project.Difficulty}")
            valid_count += 1
        except Exception as e:
            print(f"❌ {fname:<32} | Error: {e}")
            errors += 1

    print("=" * 80)
    print(f"Total Projects: {valid_count} Valid | {errors} Errors")
    return errors == 0


if __name__ == "__main__":
    success = validate_library()
    sys.exit(0 if success else 1)

