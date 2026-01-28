"""
Deploy Indian Address Parser to HuggingFace Spaces.

Usage:
    1. First login: python -c "from huggingface_hub import login; login()"
       (This opens a browser for authentication)
    2. Then run: python deploy_hf_spaces.py

This script creates a HuggingFace Space and uploads the demo files.
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def deploy():
    api = HfApi()

    # Get username
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/indian-address-parser"

    print(f"Deploying to: https://huggingface.co/spaces/{repo_id}")

    # Create Space (or get existing)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )
        print(f"Space created/confirmed: {repo_id}")
    except Exception as e:
        print(f"Error creating space: {e}")
        return

    # Prepare files to upload
    project_root = Path(__file__).parent
    demo_dir = project_root / "demo"
    src_dir = project_root / "src"

    # Create a temporary directory with the right structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Copy demo files (app.py, README.md, requirements.txt)
        for f in demo_dir.iterdir():
            if f.is_file() and not f.name.startswith("__"):
                shutil.copy2(f, tmp / f.name)

        # Copy src directory (needed for imports)
        shutil.copytree(src_dir, tmp / "src")

        # Upload to HuggingFace
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="space",
        )

    print(f"\nDeployment complete!")
    print(f"View your Space: https://huggingface.co/spaces/{repo_id}")
    print(f"\nNote: The Space runs in rules-only mode by default.")
    print(f"To enable the ML model, upload your trained model to the Space")
    print(f"and set the MODEL_PATH environment variable in Space settings.")


if __name__ == "__main__":
    deploy()
