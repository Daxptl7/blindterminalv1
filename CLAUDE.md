# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
BlindAssist is an Accessible Educational Terminal for visually impaired users, built on Raspberry Pi 5. It allows users to read text (OCR), type (Morse), interact with AI, and detect objects via tactile and audio interfaces.

## Architecture
- **Modular Structure**: The project follows a star topology. `main.py` is the orchestrator that imports and routes to all modules. Modules **must not** import each other.
- **Pi Flag Pattern**: Every module MUST include the following flags in its configuration section to abstract hardware from software:
  - `HEADLESS`: Disable GUI/Display.
  - `USE_PICAMERA`: Use Pi Camera Module 3 vs. USB Webcam.
  - `USE_GPIO`: Use physical tactile buttons vs. Keyboard simulation.
  - `USE_CORAL`: Use Google Coral Accelerator vs. CPU.
- **Centralized Audio**: All spoken output must go through `blindterminal/modules/tts.py`, which manages a non-blocking queue of speech tasks.
- **Configuration**: All constants, API keys, and thresholds are stored in `blindterminal/config/settings.json`.
- **Logging**: Every module maintains its own log file in `blindterminal/logs/`.

## Development & Testing
### Common Commands
- **Run a specific module test**: `python3 blindterminal/modules/<module_name>.py`
- **Run the main application**: `python3 blindterminal/main.py`

### Laptop Simulation Mappings
Since hardware is often unavailable, use these keyboard shortcuts to simulate physical interactions:
- **Morse Input**: `.` (Dot), `-` (Dash), `Enter` (Confirm/Send), `Backspace` (Delete), `Tab` (Space).
- **OCR/Detection**: `S` (Scan trigger), `Q` (Quit).
- **Navigation/Mode**: Number keys `1-9` for mode selection.
- **Confidential Mode**: Simulation uses `[PRIVATE]` console tags.

## Code Guidelines
- **Non-Blocking TTS**: Always use `tts.speak()` to avoid freezing the main execution loop.
- **Camera Safety**: Always wrap camera operations in `try/finally` blocks to ensure `cap.release()` is called.
- **Warmup Frames**: Discard at least 5 frames when opening a camera to allow auto-exposure to settle.
- **Error Handling**: Wrap all AI, OCR, and API calls in `try/except` blocks to prevent a single failed request from crashing the device.
- **Clean Exit**: Implement `SIGINT` handlers in every module for graceful `Ctrl+C` shutdowns.
