"""Download everything dictation.py needs into cuda_storage/. Safe to re-run; finished files are skipped."""

import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "cuda_storage"

LLAMA_BUILD = "b12527"
LLAMA_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"
LLAMA_ZIPS = [
    f"{LLAMA_BASE}/llama-{LLAMA_BUILD}-bin-win-cuda-12.4-x64.zip",
    f"{LLAMA_BASE}/cudart-llama-bin-win-cuda-12.4-x64.zip",
]
QUILL_URL = "https://huggingface.co/Quobi/Quill/resolve/main/quill-4b-Q4_K_M.gguf"


def fetch(url, dest=None):
    """Stream a URL to dest (or memory) with a simple progress line."""
    name = url.rsplit("/", 1)[-1]
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        out = open(dest, "wb") if dest else io.BytesIO()
        done = 0
        for chunk in r.iter_content(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {name}: {done * 100 // total}% of {total >> 20} MB", end="", flush=True)
        print()
        if dest:
            out.close()
            return None
        out.seek(0)
        return out


def llama_cpp():
    target = STORE / "llama.cpp"
    if (target / "llama-server.exe").exists():
        print("llama.cpp: already present")
        return
    print(f"llama.cpp {LLAMA_BUILD} (CUDA 12.4):")
    target.mkdir(parents=True, exist_ok=True)
    for url in LLAMA_ZIPS:
        with zipfile.ZipFile(fetch(url)) as z:
            z.extractall(target)


def quill():
    dest = STORE / "llm" / "quill-4b-Q4_K_M.gguf"
    if dest.exists():
        print("Quill 4B: already present")
        return
    print("Quill 4B Q4_K_M:")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    fetch(QUILL_URL, partial)
    partial.replace(dest)


def moonshine():
    from moonshine_voice.download import get_model_for_language
    from moonshine_voice.moonshine_api import ModelArch

    print("Moonshine Small Streaming:")
    path, _ = get_model_for_language("en", ModelArch.SMALL_STREAMING, cache_root=STORE / "stt")
    print(f"  {path}")


if __name__ == "__main__":
    try:
        llama_cpp()
        quill()
        moonshine()
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)
    print("\nAll models ready.")
