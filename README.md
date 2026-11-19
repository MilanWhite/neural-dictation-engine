# Neural Dictation Engine

A local, barebones Wispr Flow-style dictation bar for Windows.

- **Hold Caps Lock**: talk, then release to insert the text.
- **Double-tap Caps Lock**: hands-free. Press Caps Lock again to insert the text.
- **Right-click the bar**: quit.

The pipeline is mic → Moonshine Small Streaming (CPU) → Quill 4B Q4_K_M through llama.cpp (CUDA) → pasted into the focused text box.

## Setup (once)
Requires Windows, Python 3.12 and an NVIDIA GPU (8 GB VRAM). Double-click `setup.bat`. It creates `.venv`, installs the dependencies and runs `download_models.py`, which downloads about 3 GB of models into `cuda_storage/`.

## Run
Double-click `Start Dictation.vbs`. Caps Lock no longer toggles capitals while the app is running.

## Layout
- `dictation.py`: the whole app
- `cuda_storage/stt/`: Moonshine Small Streaming model
- `cuda_storage/llm/quill-4b-Q4_K_M.gguf`: cleanup LLM
- `cuda_storage/llama.cpp/`: llama.cpp b12527, Windows CUDA 12.4 build

## Notes
Quill sometimes answers a dictated question or request instead of cleaning it up. A guard catches this. When the output contains many words you didn't say, the app inserts the raw Moonshine transcript with filler words removed. `dictation.log` records timings and errors only, never your text.
