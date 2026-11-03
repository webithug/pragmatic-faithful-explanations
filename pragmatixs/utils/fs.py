import os
import subprocess


def get_root_dir():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get git root directory: {result.stderr}")
    return result.stdout.strip()


def load_dotenv():
    from dotenv import load_dotenv

    root_dir = get_root_dir()
    dotenv_path = os.path.join(root_dir, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        raise FileNotFoundError(f".env file not found at {dotenv_path}")
