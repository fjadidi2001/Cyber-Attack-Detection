 # OpenAvatarChat + TTS — Complete Documentation

> **From zero to hero: everything you need to know about building a real-time interactive digital human.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start (Zero to Running)](#2-quick-start-zero-to-running)
3. [Architecture — The Big Picture](#3-architecture--the-big-picture)
4. [Model Deep Dives](#4-model-deep-dives)
    - [4.1 SileroVAD (Voice Activity Detection)](#41-silerovad-voice-activity-detection)
    - [4.2 SmartTurn EOU (End-of-Utterance)](#42-smartturn-eou-end-of-utterance)
    - [4.3 SenseVoice (ASR)](#43-sensevoice-asr)
    - [4.4 OpenAI-Compatible LLM](#44-openai-compatible-llm)
    - [4.5 Qwen-Omni (End-to-End Multimodal)](#45-qwen-omni-end-to-end-multimodal)
    - [4.6 CosyVoice (TTS)](#46-cosyvoice-tts)
    - [4.7 Edge TTS](#47-edge-tts)
    - [4.8 OpenAI TTS](#48-openai-tts)
    - [4.9 LiteAvatar (2D Digital Human)](#49-liteavatar-2d-digital-human)
    - [4.10 LAM (3D Audio-to-Expression)](#410-lam-3d-audio-to-expression)
    - [4.11 MuseTalk (Video Talking Head)](#411-musetalk-video-talking-head)
    - [4.12 FlashHead (Diffusion Talking Head)](#412-flashhead-diffusion-talking-head)
    - [4.13 ChatAgent](#413-chatagent)
5. [Configuration Reference](#5-configuration-reference)
6. [Preset Modes Deep Dive](#6-preset-modes-deep-dive)
7. [Deployment Guide](#7-deployment-guide)
8. [Development & Custom Handlers](#8-development--custom-handlers)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. Project Overview

This repository contains **two related projects** in the same directory:

### 1.1 The TTS Utilities (Parent Directory)

`/home/m-sojodi/Desktop/TTS/` contains standalone Python scripts for OpenAI TTS:

| File | Purpose |
|---|---|
| `tts_openai.py` | OpenAI TTS CLI client — calls the OpenAI audio/speech API |
| `tts_handler_openai.py` | Handler module for integrating OpenAI TTS into OpenAvatarChat |
| `main.py` | Entry point that delegates to `tts_openai.main()` |
| `pyproject.toml` | Dependencies (librosa, numpy, python-dotenv, requests) |
| `config/chat_with_openai_tts.yaml` | OpenAvatarChat config that uses this TTS handler |

These are **utility scripts** — you can use them standalone:
```bash
python tts_openai.py --text "Hello world" --out output.wav --voice nova
```

Or integrate them into the full avatar system via `tts_handler_openai.py`.

### 1.2 OpenAvatarChat (Subproject)

`/home/m-sojodi/Desktop/TTS/OpenAvatarChat/` is a **full-stack interactive digital human system** (v0.6.0). It chains together:

```
[Microphone] → VAD → ASR → LLM → TTS → Avatar → [Screen/Speaker]
```

**Key capabilities:**
- Real-time voice conversation with a digital human avatar
- 14 preset configurations (different ASR/LLM/TTS/Avatar combinations)
- Half-duplex and full-duplex (interruptible) conversation modes
- WebRTC streaming for browser clients
- Web-based Gradio UI + FastAPI backend
- ChatAgent mode with tool calling, memory, and OpenClaw integration
- Runs on a single GPU-equipped PC

**Tech stack:** Python 3.11, PyTorch 2.8, ONNX Runtime, WebRTC (aiortc), Gradio 5, FastAPI, ModelScope, HuggingFace

---

## 2. Quick Start (Zero to Running)

### 2.1 Prerequisites

| Requirement | Minimum |
|---|---|
| OS | Linux (Ubuntu 22.04+ recommended) |
| Python | 3.11.7 (exact, via `.python-version`) |
| GPU | NVIDIA with CUDA 12.8+ (for local models) |
| VRAM | 8 GB+ (16 GB+ for FlashHead/MuseTalk) |
| RAM | 16 GB+ |
| Disk | 20 GB+ (for models) |
| Tools | `git-lfs`, `uv` (package manager), `build-essential` |
| API Keys | OpenAI, DashScope (or run local models) |

### 2.2 Installation

```bash
# 1. Clone with submodules (if starting fresh)
git clone --recursive <repo-url>
cd OpenAvatarChat

# 2. Set up Python environment
uv python pin 3.11.7
uv venv --python 3.11.7
source .venv/bin/activate

# 3. Install core dependencies
uv sync --extra-index-url https://download.pytorch.org/whl/cu128

# 4. Install handler-specific dependencies
python install.py --config config/chat_with_openai_compatible_bailian_cosyvoice.yaml

# 5. Configure API keys in .env
cat > .env << EOF
OPENAI_API_KEY="sk-your-key"
DASHSCOPE_API_KEY="sk-your-dashscope-key"
EOF

# 6. Download models (for local handlers)
python scripts/download_models.py --handler liteavatar
python scripts/download_models.py --handler lam
```

### 2.3 Run the Minimal Standalone TTS

```bash
cd /home/m-sojodi/Desktop/TTS
python tts_openai.py --text "Hello, welcome to the digital human system!" --out hello.wav --voice nova
```

This calls `POST https://api.openai.com/v1/audio/speech` and saves a WAV file.

### 2.4 Run the Full Avatar System

```bash
cd OpenAvatarChat
python src/demo.py --config config/chat_with_openai_compatible_bailian_cosyvoice.yaml
```

Then open `https://localhost:8282` in a browser. The UI uses WebRTC for low-latency audio/video streaming.

### 2.5 Verify It's Working

- Open the WebUI at `https://localhost:8282`
- Click "Connect" to establish a WebRTC session
- Speak into your microphone
- You should see the avatar respond with lip-synced speech

---

## 3. Architecture — The Big Picture

### 3.1 Handler Pipeline Pattern

The entire system is built around a **modular handler pipeline**. Each stage is an independent module with:

- **Well-defined I/O types** (audio in, text out, etc.)
- **Standard lifecycle** (load → create_context → handle → destroy)
- **Plugin-style discovery** (loadable from any Python path)

```
                    ┌────────────────────────────────────────────┐
                    │              ChatEngine                     │
                    │   ┌──────────┐  ┌──────────┐               │
                    │   │ Handler  │  │  Logic   │               │
                    │   │ Manager  │  │  Manager │               │
                    │   └────┬─────┘  └────┬─────┘               │
                    │        │              │                     │
                    │   ┌────▼──────────────▼─────┐               │
                    │   │     ChatSession         │               │
                    │   │  (per-client session)   │               │
                    │   └────┬──────────────┬─────┘               │
                    └────────┼──────────────┼─────────────────────┘
                             │              │
            ┌────────────────▼──┐    ┌──────▼─────────────────┐
            │   StreamManager   │    │     SignalManager      │
            │  (data streams)   │    │   (event pub/sub)      │
            └───────────────────┘    └────────────────────────┘
```

**Handler chain (half-duplex):**

```
MIC → [VAD Handler] → [ASR Handler] → [LLM Handler] → [TTS Handler] → [Avatar Handler] → SPEAKERS/SCREEN
       (audio chunks)    (text)          (text)           (audio)          (video)
```

### 3.2 Core Data Types

The system uses typed data bundles flowing through streams:

| Data Type | Direction | Description |
|---|---|---|
| `MIC_AUDIO` | Input | Raw microphone audio (48 kHz) |
| `HUMAN_AUDIO` | Internal | Downsampled/processed human audio (16 kHz) |
| `HUMAN_TEXT` | Internal | ASR transcription output |
| `AVATAR_TEXT` | Internal | LLM response text |
| `AVATAR_AUDIO` | Internal | TTS audio output (24 kHz) |
| `AVATAR_VIDEO` | Output | Rendered avatar video frames |
| `AVATAR_MOTION_DATA` | Output | Blendshape/expression coefficients |
| `CLIENT_PLAYBACK` | Output | Final audio sent to client |
| `HUMAN_VOICE_ACTIVITY` | Internal | VAD state change events |
| `EOU_TRIGGERED` | Internal | End-of-utterance signal |
| `HUMAN_DUPLEX_AUDIO` | Internal | Duplex-mode audio |
| `PERCEPTION_CONTEXT` | Internal | Agent vision system |
| `ENVIRONMENT_EVENT` | Internal | Agent environment triggers |

### 3.3 Stream Lifecycle

Each piece of data flows through a `ChatStream` with a lifecycle:

```
[STREAM_BEGIN signal] → stream_data(chunk1) → stream_data(chunk2) → ... → [STREAM_END signal]
                                                                        or
                                                                    [STREAM_CANCEL signal]
```

Key properties:
- **Cancel propagation**: When an interrupt occurs, the StreamManager cancels from leaf back to root, so all downstream handlers stop and restart
- **Ancestor tracking**: Each stream records its ancestors for targeted cancellation
- **Stream storage**: Maintains a configurable-length history for access by later handlers (e.g., SemanticTurnDetector)

### 3.4 Signal Bus

The `SignalManager` is a pub/sub event bus:

| Signal | Meaning |
|---|---|
| `SESSION_START` | A new client session began |
| `STREAM_BEGIN` | A data stream started |
| `STREAM_END` | A data stream ended normally |
| `STREAM_CANCEL` | A data stream was interrupted |
| `INTERRUPT` | User interrupted the avatar (triggered by VAD or SemanticTurn) |
| `SEMANTIC_WAIT` | LLM-determined — extend silence wait, user hasn't finished |
| `ERROR` | An error occurred in a handler |
| `SESSION_STOP` | Session ended |

### 3.5 Half-Duplex vs Full-Duplex

**Half-duplex** (default):
1. User speaks → VAD detects start → VAD detects end → ASR transcribes → LLM responds → TTS speaks → Avatar animates
2. User cannot interrupt while avatar is speaking

**Full-duplex** (via `_duplex.yaml` configs):
1. VAD runs continuously (always-on)
2. User audio is sent as `HUMAN_DUPLEX_AUDIO` type
3. SmartTurn EOU + SemanticTurnDetector decide when the user has finished their utterance
4. `SemanticTurnDetector` uses an LLM to classify: is this an interruption, a complete utterance, or should we wait?
5. Avatar speech can be interrupted mid-stream
6. Type overrides map `HUMAN_AUDIO` → `HUMAN_DUPLEX_AUDIO` and `HUMAN_TEXT` → `HUMAN_DUPLEX_TEXT`

### 3.6 Config Loading

Configuration uses **dynaconf** with YAML files. The config hierarchy:

```yaml
default:
  logger:
    log_level: "INFO"
  service:
    host: "0.0.0.0"
    port: 8282
    cert_file: "ssl_certs/localhost.crt"
    cert_key: "ssl_certs/localhost.key"
  chat_engine:
    model_root: "models"              # Where to store downloaded models
    concurrent_limit: 1               # Max concurrent sessions
    handler_search_path:              # Where to discover handler modules
      - "src/handlers"
    handler_configs:                  # Per-handler configuration
      HandlerName:
        module: "path.to.module"      # Python import path
        enabled: true                 # Optional, default true
        param1: value1               # Handler-specific params
```

---

## 4. Model Deep Dives

*How each model works — architecture, inference flow, and why it's used.*

### 4.1 SileroVAD (Voice Activity Detection)

**Purpose:** Detect when a human is speaking vs. silent.

**Architecture:**
- Pre-trained model by Silero Team (GitHub: `snakers4/silero-vad`)
- Available in ONNX and PyTorch formats
- Extremely small (~1.7 MB), runs on CPU in milliseconds
- Input: 16 kHz mono PCM audio (30ms frames)
- Output: Speech probability per frame (0.0–1.0)

**State Machine:**

```
IDLE ──[prob > threshold]──→ START ──[hold time]──→ SPEAKING
                                                        │
                                                        │ [prob < threshold]
                                                        ▼
                                                   POST_END ──[end_delay]──→ STOPPED
                                                        │
                                                        │ [prob > threshold]
                                                        ▼
                                                   SPEAKING (early-reject)
```

**Inference flow:**
1. Audio arrives in 30ms chunks (480 samples at 16 kHz)
2. Each chunk passes through the model — a small GRU-based RNN
3. If speech probability > threshold (default 0.5), state transitions
4. `start_delay` (200ms): must see consistent speech before declaring START
5. `end_delay` (1500ms): must see consistent silence before declaring STOP
6. State changes emitted as `HUMAN_VOICE_ACTIVITY` (STARTED/STOPPED)

**Configuration:**
```yaml
SileroVad:
  threshold: 0.5          # Speech probability threshold
  start_delay_ms: 200     # ms of speech before triggering
  end_delay_ms: 1500      # ms of silence before ending
  early_end_frames: 50    # Allow early end if silence persists
```

### 4.2 SmartTurn EOU (End-of-Utterance)

**Purpose:** Determine precisely when the user has finished their utterance (used in duplex mode).

**Architecture:**
- ONNX model (`smart-turn-v3.1-cpu.onnx`)
- Operates on 16 kHz audio, same stream as VAD
- Outputs: EOU probability (0.0–1.0)

**Inference flow:**
1. Runs alongside SileroVAD (only evaluates when VAD is in SPEAKING state)
2. Monitors acoustic features from the audio stream
3. When EOU probability > threshold, emits `EOU_TRIGGERED` signal
4. This triggers the pipeline to treat the accumulated audio as a complete utterance

**Why it's needed:** SileroVAD only detects speech/silence. SmartTurn EOU detects *meaningful turn completion* — a pause that indicates the user is done speaking vs. a pause mid-thought.

### 4.3 SenseVoice (ASR)

**Purpose:** Convert human speech audio to text.

**Architecture:**
- Model: `iic/SenseVoiceSmall` from ModelScope (FunASR ecosystem)
- Encoder-decoder transformer architecture
- Multilingual (Chinese, English, Cantonese, Japanese, Korean)
- Runs on CPU (optimized for real-time)
- Input: 16 kHz mono PCM, 1-second windows (16000 samples)
- Output: Text with timestamps (`start_time`, `end_time`)

**Inference flow:**
1. Audio arrives as `HUMAN_AUDIO` (16 kHz PCM chunks)
2. Accumulated in a sliding window buffer (cached overlap between chunks)
3. Each non-silent segment is fed to the model
4. Model outputs transcribed text with confidence scores
5. Text is emitted as `HUMAN_TEXT` with segment metadata

**Configuration:**
```yaml
SenseVoice:
  model_dir: "models/SenseVoiceSmall"
  use_vad: true          # Use built-in VAD for segmentation
  sample_rate: 16000
  language: "auto"       # Auto-detect language
```

### 4.4 OpenAI-Compatible LLM

**Purpose:** Generate conversational responses from the ASR transcript.

**Architecture:**
- Generic OpenAI SDK client — works with any OpenAI-compatible API
- Supports: OpenAI, DashScope (Qwen), Ollama (local), Gemini, vLLM, etc.
- Modal: text-only or multimodal (text + base64-encoded images)
- Streaming: server-sent events for incremental text

**Inference flow:**
1. Receives `HUMAN_TEXT` from ASR (or via `HUMAN_DUPLEX_TEXT` in duplex mode)
2. Formats chat history (system prompt + conversation turns)
3. Sends to the API endpoint
4. Receives streaming response tokens
5. Emits each token as `AVATAR_TEXT` chunks
6. At stream end, signals completion

**Configuration:**
```yaml
OpenAICompatible:
  api_key: "${OPENAI_API_KEY}"    # Environment variable reference
  api_url: "https://api.openai.com/v1"  # or Ollama, DashScope, etc.
  model_name: "gpt-4o-mini"
  system_prompt: "You are a helpful assistant."
  temperature: 0.7
  max_tokens: 512
  multimodal: false               # Enable vision capabilities
```

### 4.5 Qwen-Omni (End-to-End Multimodal)

**Purpose:** Bypass the ASR + LLM + TTS pipeline with a single end-to-end model.

**Architecture:**
- DashScope WebSocket API (`qwen-omni-turbo-realtime`)
- Handles audio input directly (no separate ASR needed)
- Can output both text and audio simultaneously
- Built-in turn detection (decides when user has finished speaking)
- Three concurrent threads per session: input sender, result receiver, heartbeat keepalive

**How it integrates:**
1. Receives raw audio chunks via WebSocket
2. Sends them to DashScope incrementally
3. Receives back text + audio responses
4. Text goes to `AVATAR_TEXT` (for display/chat history)
5. Audio goes directly to `AVATAR_AUDIO` (bypassing TTS)
6. Turn detection events map to `EOU_TRIGGERED`

**Tradeoffs:**
- Pros: Lowest latency (bypasses 3 separate models), built-in turn detection, multimodal
- Cons: Locked to DashScope API, less control over individual components

### 4.6 CosyVoice (TTS)

**Purpose:** Convert LLM response text into natural speech with voice cloning.

**Architecture:**
- Model: Alibaba's CosyVoice from ModelScope
- Flow-matching based text-to-speech
- Supports zero-shot voice cloning (reference audio + text)
- Runs in a **spawned GPU subprocess** via `torch.multiprocessing.Process`
- Input: Text string
- Output: 24 kHz mono PCM audio

**Why subprocess?** GPU inference blocks the Python GIL and async event loop. By spawning a separate process, the main handler can continue processing while the GPU works.

**Inference flow:**
```
Handler thread:                    GPU subprocess:
  receive AVATAR_TEXT              wait on queue
  → send to inference queue  ──→  → load CosyVoice model
  → return immediately             → run flow-matching decoder
                                   → return audio bytes via queue
  ← receive AVATAR_AUDIO  ←──────
  → submit to stream
```

**Voice cloning:**
1. Provide a reference audio file (3–10 seconds of clean speech)
2. Provide the reference text (what's spoken in the audio)
3. CosyVoice extracts the speaker embedding
4. New text is synthesized in that voice

**Configuration:**
```yaml
CosyVoice:
  model_dir: "models/CosyVoice-300M"
  voice: "default"               # Or use reference_audio for cloning
  reference_audio: "voices/speaker.wav"
  reference_text: "The reference text."
  sample_rate: 24000
  speed: 1.0
```

### 4.7 Edge TTS

**Purpose:** Free, lightweight text-to-speech with no GPU needed.

**Architecture:**
- Python async wrapper around Microsoft Edge's cloud TTS service
- Uses `edge-tts` library
- 100+ voices across 50+ languages
- No API key, no registration, no cost

**Inference flow:**
1. Receives `AVATAR_TEXT`
2. Calls `edge-tts` async with selected voice
3. Receives MP3 audio stream
4. Decodes to PCM and emits as `AVATAR_AUDIO` (24 kHz)

**Limitations:**
- Cloud-dependent (requires internet)
- No voice cloning
- ~1–2 second latency per utterance
- May have usage limits (though undocumented)

### 4.8 OpenAI TTS

**Purpose:** High-quality cloud TTS with voice style control.

**Architecture:**
- HTTP POST to `https://api.openai.com/v1/audio/speech`
- Models: `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts-2025-12-15`
- Voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- The `gpt-4o-mini` model supports `instructions` parameter for voice style/tone

**Inference flow:**
1. Receives `AVATAR_TEXT`
2. Optionally splits text into sentences for streaming
3. Makes HTTP POST with the text chunk
4. Receives WAV/MP3 audio bytes
5. Converts via librosa to float32 numpy array (24 kHz)
6. Emits as `AVATAR_AUDIO` (chunk by chunk for sentence-level streaming)
7. On stream end, emits zero-padding finish signal

**Streaming with chunked mode:**
```python
sentences = split_sentences(text)  # Split on . ! ?
for sentence in sentences:
    audio = openai_tts(sentence, ...)  # One API call per sentence
    stream_data(audio)                 # Stream immediately
```

**Standalone usage** (`/home/m-sojodi/Desktop/TTS/tts_openai.py`):
```bash
python tts_openai.py --text "Hello world" --out output.wav --voice nova
```

### 4.9 LiteAvatar (2D Digital Human)

**Purpose:** Render a 2D digital human avatar with lip-sync.

**Architecture:**
- CPU-based inference pipeline
- 100+ character avatars from ModelScope (`HumanAIGC-Engineering/LiteAvatarGallery`)
- Output: 25 FPS video frames
- Components:
  - **Audio2Signal**: Speed-limited audio processing (matches audio clock)
  - **Tts2faceCpuAdapter**: Lip-sync neural network (audio → mouth shape)
  - **Video compositor**: Blends mouth shape onto background avatar image
  - **Frame metronome**: Maintains 25 FPS output rate

**Inference flow:**
```
Audio in → Audio2Signal (throttled to real-time) → Tts2faceCpuAdapter → mouth shape
                                                                              ↓
Background image ←───────────────────────────────────────────────────→ Compositor
                                                                              ↓
                                                                     Video frame (25 FPS)
```

**Threading:** A worker pool manages multiple avatar instances. Each worker uses shared memory buffers for zero-copy data transfer between processes.

**Configuration:**
```yaml
LiteAvatar:
  avatar_id: "default"          # Which character from the gallery
  fps: 25
  worker_count: 2               # Pool size for concurrent processing
  enable_audio_playback: true
  sample_rate: 24000
```

### 4.10 LAM (3D Audio-to-Expression)

**Purpose:** Generate 3D facial expression coefficients from audio (for 3D avatars).

**Architecture:**
- Two-stage pipeline:
  1. **wav2vec2-base-960h** (HuggingFace): Audio feature extraction
  2. **LAM_audio2exp_streaming** (ModelScope): Expression coefficient prediction
- Output: Expression coefficients (blendshape weights) — no video rendering
- Runs on CPU
- Input: 16 kHz mono audio
- Output: Array of expression coefficients (e.g., 50+ blendshapes)

**Inference flow:**
```
Audio chunks → wav2vec2 → feature vectors → LAM model → expression coefficients
                                                             ↓
                                                    AVATAR_MOTION_DATA
                                                    (to 3D renderer)
```

**Why wav2vec2?** wav2vec2 is a self-supervised speech representation model from Facebook AI. It converts raw audio into rich feature vectors that capture phonetic content, prosody, and speaker characteristics — exactly what's needed for expression prediction.

### 4.11 MuseTalk (Video Talking Head)

**Purpose:** Realistic video-driven talking head avatar.

**Architecture:**
- Multi-model pipeline running on GPU:
  1. **WhisperModel** (small): Audio → speech embeddings
  2. **UNet**: Speech embeddings → face landmark generation
  3. **SAF** (Segment Anything / Face parsing): Separate face from background
  4. **Compositor**: Blend generated face onto original video
- Input: Reference video + live audio
- Output: 512×512 video frames at ~30 FPS

**Pipeline (4 threads, 4 queues):**

```
Thread 1 (Audio):   Audio chunks → WhisperModel → speech embeddings → Queue 1
                                                              ↓
Thread 2 (UNet):    Queue 1 → UNet face generation → face frame → Queue 2
                                                                      ↓
Thread 3 (Compose): Queue 2 → SAF face parsing → background compositing → Queue 3
                                                                              ↓
Thread 4 (Output):  Queue 3 → AVATAR_VIDEO frames → stream output
```

**Key detail — whisper embeddings:** The Whisper speech recognition model's internal encoder representations are used, not its text output. These embeddings capture phonetic and prosodic information needed for lip-sync.

**Inference flow (per chunk):**
1. Audio chunk passed to Whisper encoder → 512-dim embedding vector
2. Embedding fed to UNet along with previous face frame → new face landmarks
3. SAF segments face region from background in the reference video frame
4. Compositor warps the reference face to match predicted landmarks
5. Result blended with background → final video frame

### 4.12 FlashHead (Diffusion Talking Head)

**Purpose:** State-of-the-art diffusion-based talking head generation.

**Architecture:**
- Model: SoulX-FlashHead (SoulX-FlashHead-1_3B)
- Based on **DiT** (Diffusion Transformer) — a transformer-based diffusion model
- Uses **flow-matching** for faster inference than traditional diffusion
- Additional components:
  - **wav2vec2**: Audio feature extraction
  - **VAE** (Variational Autoencoder): Image latent encoding/decoding
  - **Sliding audio window**: 8 seconds of context for temporal consistency
  - **Motion-frame latents**: Previous frame latents for smooth video output
- Output: 512×512 video frames

**Why DiT?** Traditional diffusion models use UNet backbones. DiT replaces the UNet with a transformer, which scales better with compute and produces higher-quality results. FlashHead uses DiT for state-of-the-art face generation quality.

**Inference flow:**

```
Sliding audio buffer (8s) → wav2vec2 → audio features
                                                    ↓
Previous frame latent (motion-frame) ──→ DiT denoising (flow-matching)
                                                    ↓
                                            VAE decoder
                                                    ↓
                                         512×512 video frame
```

**Pipeline modes:**
- **Synchronous**: One frame at a time (simpler, lower throughput)
- **Pipelined**: Overlap audio processing with denoising for higher FPS

**Configuration:**
```yaml
FlashHead:
  model_type: "soulx-lite"
  fps: 25
  face_crop_size: 512
  audio_window_seconds: 8       # How much context to keep
  pipeline_mode: "pipelined"    # or "synchronous"
```

### 4.13 ChatAgent

**Purpose:** Replace the simple LLM handler with a full agent system with tool calling, memory, perception, and OpenClaw integration.

**Architecture — 4 main subsystems:**

#### 4.13.1 PromptCompiler (4-Layer Prompt)

```
Layer 1: Stable Core (system_message)
  - Rules, constraints, agent identity (fixed per session)

Layer 2: Persona Snapshot (system_message)
  - Personality, background, knowledge (from OpenClaw, periodically refreshed)

Layer 3: Environment State (system_message)
  - <current_time>, <user_data>, <scene_info> (tag-wrapped, updated per turn)

Layer 4: Recent Dialogue (messages)
  - History + perception observations + tool call results
```

#### 4.13.2 SessionMemoryManager

| Component | Function |
|---|---|
| **WorkingMemory** | Recent dialogue turns (15 default), auto-compacts via LLM when full |
| **PerceptionBuffer** | Vision events with time decay, TTL, category-based aggregation |
| **SessionSummary** | Rule-based intent/topic extraction |
| **WriteBackQueue** | Async persistence of important events to OpenClaw |

**Auto-compact flow:**
```
WorkingMemory hits threshold (15 turns)
→ Save full transcript to WriteBackQueue
→ LLM compresses: "The user asked about X, then Y, agent responded Z"
→ Keep last N turns (5 default) + compressed summary
→ Rehydrate task brief and environment state
```

#### 4.13.3 ToolRegistry

Central tool management with schema caching and execution dispatch:

```python
class ToolRegistry:
    def register_tool(self, tool: BaseTool): ...
    def get_openai_tools_schema(self) -> list[dict]: ...
    async def execute_tool(self, name: str, args: dict) -> str: ...
```

Built-in tools:
- **GetCurrentTimeTool** — Returns current date/time
- **GetSystemInfoTool** — Returns system information
- **SpawnAgentTool** — Spawns sub-agents (explore, analyst, oc_delegate)
- **ExecApproveTool** — Approve/deny background command execution
- **PendingConfirmationsTool** — Manage pending confirmation items

**SpawnAgent types:**
| Type | Backend | Use |
|---|---|---|
| `explore` | Local (qwen-turbo) | Read-only file search |
| `analyst` | Local (qwen-plus) | Multi-step analysis |
| `oc_delegate` | OpenClaw | Complex tasks via MCP |

#### 4.13.4 OC Bridge (OpenClaw Integration)

Two-way communication with the OpenClaw backend:

```
┌────────────────┐     HTTP/MCP      ┌─────────────────┐
│  ChatAgent     │ ◄──────────────►  │   OpenClaw      │
│  (this system) │                   │  (control plane) │
└────────────────┘                   └─────────────────┘
```

Components:
- **PersonaSnapshotManager**: Periodically fetches persona from OC
- **TaskNotificationQueue**: Receives async task results from OC
- **TaskMirror**: File-based task state sync
- **PendingConfirmationsManager**: Manages human-in-the-loop approval flow
- **OcChannelClient**: HTTP sync/async requests to OC
- **OcMcpClient**: Background MCP thread for tool discovery

---

## 5. Configuration Reference

### 5.1 Global Config

```yaml
default:
  logger:
    log_level: "INFO"               # DEBUG, INFO, WARNING, ERROR
  service:
    host: "0.0.0.0"                # Listen address
    port: 8282                     # HTTPS port
    cert_file: "ssl_certs/localhost.crt"
    cert_key: "ssl_certs/localhost.key"
  history:                          # Optional (for duplex mode)
    retention_mode: "by_both"       # "by_user", "by_agent", "by_both"
    max_events: 1000
    max_age_seconds: 3600
    cleanup_interval_seconds: 30
  chat_engine:
    model_root: "models"            # Model storage directory
    concurrent_limit: 1             # Max parallel sessions
    handler_search_path:            # Handler discovery paths
      - "src/handlers"
      - "/home/m-sojodi/Desktop/TTS"  # Custom path for external handlers
```

### 5.2 Handler Config Structure

Each handler under `chat_engine.handler_configs` follows:

```yaml
HandlerName:
  module: "path.to.module"          # Python module path
  enabled: true                     # Optional, default true
  # ...handler-specific parameters...
```

### 5.3 Handler-Specific Parameters

#### VAD Handlers

**SileroVAD:**
```yaml
SileroVad:
  threshold: 0.5
  start_delay_ms: 200
  end_delay_ms: 1500
  early_end_frames: 50
  use_onnx: true
```

**SmartTurn EOU (duplex only):**
```yaml
SmartTurnEOU:
  model_path: "models/smart-turn-v3.1-cpu/smart-turn-v3.1-cpu.onnx"
  threshold: 0.3
  chunk_size: 480                 # 30ms at 16kHz
```

#### ASR Handlers

**SenseVoice:**
```yaml
SenseVoice:
  model_dir: "models/SenseVoiceSmall"
  use_vad: true
  sample_rate: 16000
  language: "auto"
  max_single_segment_time: 30000  # ms
```

**Bailian ASR:**
```yaml
BailianASR:
  api_key: "${DASHSCOPE_API_KEY}"
  sample_rate: 16000
  enable_semantic_punctuation: true
  language_hints: ["zh", "en"]
```

#### LLM Handlers

**OpenAICompatible:**
```yaml
OpenAICompatible:
  api_key: "${OPENAI_API_KEY}"
  api_url: "https://api.openai.com/v1"
  model_name: "gpt-4o-mini"
  system_prompt: "You are a helpful AI assistant."
  temperature: 0.7
  max_tokens: 512
  multimodal: false
  multimodal_camera: false
  text_filter_pattern: "[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]"
```

**QwenOmni:**
```yaml
QwenOmni:
  api_key: "${DASHSCOPE_API_KEY}"
  model: "qwen-omni-turbo-realtime"
  sample_rate: 24000
  enable_turn_detection: true
```

**Dify:**
```yaml
Dify:
  api_key: "${DIFY_API_KEY}"
  api_url: "https://api.dify.ai/v1"
  user: "open-avatar-chat"
  response_mode: "streaming"        # or "blocking"
  poll_interval: 0.5
```

#### TTS Handlers

**BailianCosyVoice:**
```yaml
BailianCosyVoice:
  api_key: "${DASHSCOPE_API_KEY}"
  model: "cosyvoice-v1"
  voice: "longxiaochun"            # or "longwan", "luna", "shanshan", etc.
  sample_rate: 24000
  format: "wav"
```

**CosyVoiceLocal:**
```yaml
CosyVoiceLocal:
  model_dir: "models/CosyVoice-300M"
  voice: "default"
  reference_audio: null            # Path to reference audio for cloning
  reference_text: null             # Transcription of reference audio
  sample_rate: 24000
  speed: 1.0
```

**EdgeTTS:**
```yaml
EdgeTTS:
  voice: "zh-CN-XiaoxiaoNeural"   # See edge-tts --list-voices
  sample_rate: 24000
  volume: "+0%"
  rate: "+0%"
  pitch: "+0Hz"
```

**OpenAI_TTS:**
```yaml
OpenAI_TTS:
  api_key: "${OPENAI_API_KEY}"
  voice: "nova"
  model: "gpt-4o-mini-tts-2025-12-15"
  instructions: "Speak naturally and clearly."
  sample_rate: 24000
```

#### Avatar Handlers

**LiteAvatar:**
```yaml
LiteAvatar:
  avatar_id: "xiaoxiaoke"
  fps: 25
  worker_count: 2
  sample_rate: 24000
```

**LAM:**
```yaml
LAM:
  model_path: "models/LAM_audio2exp_streaming"
  wav2vec2_path: "models/wav2vec2-base-960h"
  sample_rate: 16000
  fps: 30
```

**MuseTalk:**
```yaml
MuseTalk:
  model_path: "models/musetalk"
  reference_video: "resource/reference.mp4"
  fps: 30
  batch_size: 2
  sample_rate: 16000
```

**FlashHead:**
```yaml
FlashHead:
  model_type: "soulx-lite"
  ckpt_dir: "models/SoulX-FlashHead-1_3B"
  fps: 25
  face_crop_size: 512
  audio_window_seconds: 8
  pipeline_mode: "pipelined"
```

---

## 6. Preset Modes Deep Dive

All 14 preset configs are in `config/`. Here's what each one does:

### 6.1 Simple Modes (Cloud APIs)

| Config | ASR | LLM | TTS | Avatar | Best For |
|---|---|---|---|---|---|
| `chat_with_lam.yaml` | SenseVoice | API | API | LAM | Quickest 3D demo |
| `chat_with_openai_compatible_bailian_cosyvoice.yaml` | SenseVoice | API | Bailian CosyVoice | LiteAvatar | **Recommended starting point** |
| `chat_with_openai_compatible_edge_tts.yaml` | SenseVoice | API | Edge TTS | LiteAvatar | Free TTS, no API key needed |
| `chat_with_openai_tts.yaml` | SenseVoice | API | OpenAI TTS | LiteAvatar | High-quality TTS |

### 6.2 Local Model Modes

| Config | ASR | LLM | TTS | Avatar | Best For |
|---|---|---|---|---|---|
| `chat_with_openai_compatible.yaml` | SenseVoice | API | CosyVoice Local | LiteAvatar | Offline TTS (GPU required) |

### 6.3 Multimodal Mode

| Config | Model | Best For |
|---|---|---|
| `chat_with_qwen_omni.yaml` | Qwen-Omni (end-to-end) | Lowest latency, vision-capable |

Qwen-Omni bypasses the entire ASR → LLM → TTS chain. It takes audio directly and outputs both text and audio. This gives the lowest possible end-to-end latency but locks you to DashScope.

### 6.4 Duplex Modes

| Config | Changes from base | Best For |
|---|---|---|
| `chat_with_lam_duplex.yaml` | LAM + duplex | Interruptible 3D avatar |
| `chat_with_openai_compatible_bailian_cosyvoice_duplex.yaml` | LiteAvatar + duplex | Interruptible 2D avatar |
| `chat_with_openai_compatible_bailian_cosyvoice_flashhead_duplex.yaml` | FlashHead + duplex | Interruptible high-quality video |
| `chat_with_openai_compatible_bailian_cosyvoice_musetalk_duplex.yaml` | MuseTalk + duplex | Interruptible video-driven |

Duplex configs add:
- **DuplexVAD handler** — always-on VAD, outputs `HUMAN_DUPLEX_AUDIO`
- **SmartTurnEOU handler** — end-of-utterance detection
- **SemanticTurnDetector** — LLM-based arbitration for interruption detection
- **Type overrides**: `HUMAN_AUDIO` → `HUMAN_DUPLEX_AUDIO`, `HUMAN_TEXT` → `HUMAN_DUPLEX_TEXT`

### 6.5 FlashHead Modes

| Config | Avatar | Best For |
|---|---|---|
| `chat_with_openai_compatible_bailian_cosyvoice_flashhead.yaml` | FlashHead | Best quality video |
| `chat_with_openai_compatible_bailian_cosyvoice_flashhead_duplex.yaml` | FlashHead + duplex | Best quality + interruptible |
| `chat_with_openai_compatible_bailian_cosyvoice_flashhead_duplex_agent.yaml` | FlashHead + duplex + agent | Full-featured beta |

### 6.6 The Ultimate Mode

`chat_with_openai_compatible_bailian_cosyvoice_flashhead_duplex_agent.yaml`:
- FlashHead (best video quality)
- Duplex (interruptible conversation)
- ChatAgent (tool calling, memory, perception, OC integration)

This is the most feature-complete but also the most demanding.

---

## 7. Deployment Guide

### 7.1 SSL Certificates

WebRTC requires HTTPS. For local development, use self-signed certs:

```bash
# Generate self-signed cert
mkdir -p ssl_certs
openssl req -x509 -newkey rsa:4096 \
  -keyout ssl_certs/localhost.key \
  -out ssl_certs/localhost.crt \
  -days 365 -nodes \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```

For production with a public IP/domain, use Let's Encrypt:

```bash
sudo certbot certonly --standalone -d yourdomain.com
# Copy to ssl_certs/ and update config
```

### 7.2 TURN Server

WebRTC often needs a TURN server for NAT traversal. The project supports **coturn**:

```bash
# Using Docker (recommended)
docker-compose -f docker-compose.yml up coturn -d
```

Or install natively:
```bash
sudo apt install coturn
sudo coturn -n \
  --realm=openavatarchat \
  --listening-port=3478 \
  --tls-listening-port=5349 \
  --fingerprint \
  --lt-cred-mech \
  --user=username:password \
  --cert=ssl_certs/localhost.crt \
  --pkey=ssl_certs/localhost.key
```

TURN providers supported: Twilio (cloud), coturn (self-hosted).

### 7.3 VRAM Requirements

| Components | Min VRAM | Recommended VRAM |
|---|---|---|
| SenseVoice + LiteAvatar (cloud LLM/TTS) | 4 GB | 8 GB |
| + CosyVoice Local | 8 GB | 12 GB |
| + MuseTalk | 12 GB | 16 GB |
| + FlashHead | 16 GB | 24 GB |
| + FlashHead + Agent | 24 GB | 32 GB |

### 7.4 Docker Deployment

```bash
# Build and run with Docker
cd OpenAvatarChat
docker build -t openavatarchat .
docker run --gpus all -p 8282:8282 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/.env:/app/.env \
  openavatarchat
```

---

## 8. Development & Custom Handlers

### 8.1 Handler Lifecycle

Every handler inherits from `HandlerBase` and follows this lifecycle:

```
1. load()
   ─ Model loading, GPU allocation, file checks

2. create_context()
   ─ Per-session state initialization

3. warmup_context()
   ─ Optional: pre-warm models, allocate buffers

4. start_context()
   ─ Optional: start background threads

5. handle()             ◄── Called repeatedly with each input
   ─ Process input, emit output

6. destroy_context()
   ─ Clean up per-session state

7. destroy()
   ─ Release global resources
```

### 8.2 Writing a Custom Handler

```python
from chat_engine.common.handler_base import HandlerBase
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_data_type import ChatDataType

class MyCustomHandler(HandlerBase):
    def get_handler_info(self):
        return {
            "name": "MyCustomHandler",
            "description": "Does something awesome",
        }

    def load(self, handler_env, handler_config):
        # Load models, allocate resources
        self.model = load_my_model(handler_config)

    def create_context(self, session_context):
        return MyContext(session_context)

    async def handle(self, context, chat_data):
        # Process input
        input_text = chat_data.get_data()
        # Do something
        result = self.model.process(input_text)
        # Emit output
        await context.submit_data(
            ChatDataType.AVATAR_TEXT,
            result,
        )

    def destroy(self):
        # Release GPU memory, close files
        pass
```

### 8.3 Registering the Handler

1. Place the module file in a directory under `handler_search_path`
2. Add it to your YAML config:
```yaml
handler_configs:
  MyCustomHandler:
    module: mypackage.my_handler
    param1: value1
```
3. Wire it into the pipeline by setting its input/output types

### 8.4 Testing

```bash
# Unit tests (if written)
pytest tests/

# End-to-end test with a specific config
python src/demo.py --config config/my_test_config.yaml
```

---

## 9. Troubleshooting & FAQ

### 9.1 Common Issues

| Symptom | Likely Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named '...'` | missing handler dependency | Run `python install.py --config your_config.yaml` |
| CUDA out of memory | VRAM exceeded | Use a simpler avatar model, reduce batch_size |
| WebRTC connection fails | No TURN server / SSL issue | Set up coturn or use `--no-ssl` for local dev |
| Avatar not moving | Missing model files | Run `python scripts/download_models.py --handler <name>` |
| High latency (>5s) | Cloud API latency or GPU contention | Use faster models, check internet, reduce concurrent sessions |
| Audio crackling/popping | Sample rate mismatch | Check all handlers use consistent sample rates (16k/24k) |
| ASR returns nothing | VAD not triggering or mic not working | Check mic permissions, adjust VAD threshold |
| TTS cuts off early | Stream cancelled by interrupt | Check duplex config, adjust end_delay |

### 9.2 Debugging Tips

**Increase log level:**
```yaml
logger:
  log_level: "DEBUG"
```

**Use the DataTool WebSocket monitor:**
```yaml
handler_configs:
  DataTool:
    module: manager/handler_data_tool
```
Then connect to `wss://localhost:8282/ws/datatool` to see all data flowing through the system in real-time.

**Check model files:**
```bash
ls -la models/  # Should have downloaded model directories
```

**Test components individually:**
```bash
# Test TTS alone
python /home/m-sojodi/Desktop/TTS/tts_openai.py --text "Test" --out test.wav

# Test ASR alone
python scripts/test_asr.py --audio test.wav
```

### 9.3 Performance Tuning

| Goal | Change |
|---|---|
| Lower latency | Use Qwen-Omni (end-to-end), Edge TTS (no GPU), disable duplex |
| Better quality | Use FlashHead avatar, CosyVoice TTS, GPT-4o LLM |
| Lower VRAM | Use LiteAvatar instead of FlashHead, Edge TTS instead of CosyVoice local |
| Run on CPU | LiteAvatar, SileroVAD, SenseVoice all support CPU. Use cloud LLM/TTS. |
| Production ready | Set up SSL certs, TURN server, systemd service for auto-restart |

### 9.4 API Key Setup

```bash
# Create/edit .env file in OpenAvatarChat/
cat > .env << EOF
OPENAI_API_KEY="sk-..."
DASHSCOPE_API_KEY="sk-..."
GEMINI_API_KEY="AIza..."
EOF
```

Environment variables use `${VAR_NAME}` syntax in YAML configs.

---

## Appendix: Directory Reference

```
/home/m-sojodi/Desktop/TTS/
├── .env                          # API keys for parent TTS project
├── main.py                       # Entry point → tts_openai.main()
├── tts_openai.py                 # OpenAI TTS CLI + library
├── tts_handler_openai.py         # OpenAvatarChat handler for OpenAI TTS
├── pyproject.toml                # Dependencies
├── config/
│   └── chat_with_openai_tts.yaml # Config using this TTS handler
├── output.wav                    # Generated TTS output
├── README.md
└── docs.md                       # This file

/home/m-sojodi/Desktop/TTS/OpenAvatarChat/
├── .env                          # API keys
├── install.py                    # One-stop dependency installer
├── pyproject.toml                # Project config + dependencies
├── src/
│   ├── demo.py                   # Main entry point
│   ├── chat_engine/              # Core engine
│   │   ├── chat_engine.py        # Engine orchestrator
│   │   ├── core/                 # Core runtime (session, streams, signals)
│   │   ├── contexts/             # Session/Handler contexts
│   │   ├── common/               # Base classes (HandlerBase, LogicBase)
│   │   └── data_models/          # Data types, signals, streams
│   ├── handlers/                 # All handler modules
│   │   ├── vad/                  # SileroVAD, SmartTurnEOU
│   │   ├── asr/                  # SenseVoice, BailianASR
│   │   ├── llm/                  # OpenAICompatible, QwenOmni, Dify
│   │   ├── tts/                  # CosyVoice, Bailian TTS, EdgeTTS
│   │   ├── avatar/               # LiteAvatar, LAM, MuseTalk, FlashHead
│   │   ├── agent/                # ChatAgent
│   │   └── manager/              # DataTool console
│   ├── engine_utils/             # Utilities (audio, import, paths)
│   └── service/                  # FastAPI, Gradio, RTC, logging
├── config/                       # 14 YAML preset configs
├── scripts/                      # Model downloaders
├── docs/                         # Built-in documentation
├── models/                       # Downloaded model files
├── resource/                     # Avatar resources
└── extensions/                   # OpenClaw bridge
```
