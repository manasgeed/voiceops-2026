"""Script to create the official submission archive team_name_voiceops.zip.

Strictly packages required submission artifacts:
- README.md
- .env.example
- requirements.txt
- run.py
- src/ (clean python sources without __pycache__)
- report.pdf
- sample_outputs/
- DEMO_VIDEO_SCRIPT.md

Strictly excludes:
- .env (live secrets)
- dataset/ or supplied data/ CSVs
- __pycache__ and *.pyc
- .git and .pytest_cache
"""
import os
import zipfile
from pathlib import Path


def create_submission_zip(output_zip: str = "team_name_voiceops.zip"):
    root = Path(".")
    zip_path = Path(output_zip)
    if zip_path.exists():
        zip_path.unlink()

    include_patterns = [
        "README.md",
        ".env.example",
        "requirements.txt",
        "run.py",
        "report.pdf",
        "DEMO_VIDEO_SCRIPT.md",
        "FINAL_REPORT.md",
    ]

    include_dirs = [
        "src",
        "sample_outputs",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for pattern in include_patterns:
            p = root / pattern
            if p.exists():
                print(f"Adding file: {pattern}")
                zf.write(p, arcname=pattern)
            else:
                print(f"Warning: file not found: {pattern}")

        # Add directories
        for d in include_dirs:
            dir_path = root / d
            if not dir_path.exists():
                print(f"Warning: dir not found: {d}")
                continue
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    # Check exclusions
                    if "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
                        continue
                    if ".env" in file_path.name:
                        continue
                    arcname = str(file_path.relative_to(root)).replace("\\", "/")
                    print(f"Adding: {arcname}")
                    zf.write(file_path, arcname=arcname)

    print(f"\nSuccessfully generated {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_submission_zip("team_name_voiceops.zip")
