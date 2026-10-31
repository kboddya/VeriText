# VeriText — Modern AI Text & Document Authenticity Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**VeriText** is a high-performance, local AI-generated text and document detector with bilingual support (Russian & English). It accurately identifies synthetic content produced by modern Large Language Models — including **ChatGPT (GPT-4o, o1, o3)**, **DeepSeek (V3, R1)**, **Claude 3.5**, **GigaChat**, **YandexGPT**, **Qwen 2.5**, and **Llama 3**.

Available as both a **FastAPI-powered Web Application** (with paragraph-level heatmaps) and a high-throughput **Command-Line Interface (CLI)** for documents and batch datasets.

---

## 🚀 Key Features

* **Dual Hybrid Scoring Architecture:**
  * **Macro Context Score:** Evaluates full-text continuous multi-window attention (480-token windows with a 240-token stride) to capture overarching narrative consistency.
  * **Micro Paragraph-Weighted Score:** Segments documents into paragraphs and calculates word-count-weighted probabilities to detect localized synthetic inserts (*cyborg text*).
  * **Dual Gauge Visualizations:** Side-by-side transparent metrics on the Web UI comparing macro vs micro predictions.
* **Unified Web Workspace & Sorting:**
  * Streamlined single-page interface accepting direct pasted text or attached files (`.pdf`, `.docx`, `.txt`, `.md`) via drag-and-drop.
  * Interactive paragraph analysis cards with real-time filtering and 3 sorting modes: **Original Text Order**, **Highest AI % First (Descending)**, and **Lowest AI % First (Ascending)**.
  * Real-time in-flight inference cancellation (*Abort / Stop* support).
* **SOTA Language-Specific Transformer Encoders:**
  * **Russian (`ru`):** `toiletsandpaper/russian-ai-text-detector-bert` (fine-tuned on the `LLMTrace` benchmark: GPT-4o, DeepSeek, GigaChat, YandexGPT, Qwen, o1).
  * **English (`en`):** `mdrakibali/deberta-ai-detector-v3` (powered by `microsoft/deberta-v3-large`: ChatGPT, Claude 3.5, Gemini, Llama 3).
* **Canonical Preprocessing & Normalization:**
  * Eliminates typographical false positives on human text (converts `ё → е`, normalizes guillemets `«» → "`, em-dashes `— → -`, and ellipses `… → ...`).
  * Strips Markdown formatting artifacts (`#`, `**`, `-`, `>`, tables, code fences) and zero-width characters that otherwise bias transformer tokenizers.
* **Adaptive Sliding-Window Chunking for Long Texts:**
  * Extends analysis beyond the standard 512-token limit via overlapping stride windows (480 tokens with a 240-token stride).
  * Blends average confidence with peak anomaly scoring to uncover localized AI-generated insertions.
* **Weighted Multi-Format Document Scoring:**
  * Inspects **PDF (`.pdf`)**, **Microsoft Word (`.docx`)**, **Markdown (`.md`)**, and **Plain Text (`.txt`)** files.
  * Calculates an authentic, word-count-weighted overall score rather than naive paragraph averaging.
* **Batch Tabular Dataset Scoring:**
  * Efficient batch inference for **CSV** and **TSV** datasets with progress streaming and export.
* **Generative Cliché & Marker Inspection:**
  * Flags frequent RLHF and synthetic styling patterns (*"it is worth noting"*, *"delve into"*, *"важно отметить"*, *"играет ключевую роль"*).
* **Autonomous Docker Environment:**
  * Two-tier isolated architecture: Nginx reverse proxy + FastAPI backend.
  * Model weights are preloaded during the Docker build stage for instantaneous startup without runtime downloads.
* **Hardware Acceleration:**
  * Native support for **NVIDIA GPUs** (CUDA), **Apple Silicon** (Metal / MPS), and optimized multi-core **ARM64 / x86_64 CPUs**.

---

## 📁 Repository Structure

```text
veritext/
├── backend/                      # Python package: core ML engine, parsers, CLI & API
│   ├── __init__.py               # Package exports
│   ├── __main__.py               # Entrypoint for python -m backend
│   ├── config.py                 # Device selection & model catalog
│   ├── cancellation.py           # In-flight task tracking & cancellation manager
│   ├── engine.py                 # Transformer inference engine & sliding-window logic
│   ├── preprocessing.py          # Canonical text normalization & language detection
│   ├── markers.py                # Generative stylistic marker detection
│   ├── parsers.py                # Document extractors (PDF, DOCX, MD, TXT)
│   ├── cli.py                    # Command-line interface (`veritext`)
│   └── api/                      # REST API application
│       ├── __init__.py
│       ├── app.py                # FastAPI application factory & lifespan
│       ├── routes.py             # HTTP routes (/api/health, /api/score, /api/abort, etc.)
│       └── schemas.py            # Pydantic request models
├── web/                          # Web client & proxy
│   ├── nginx.conf                # Nginx reverse proxy configuration
│   └── public/                   # Static Web UI assets
│       ├── index.html            # Unified hybrid single-page interface
│       ├── css/style.css         # Custom styling
│       └── js/app.js             # Client application logic with sorting & cancellation
├── docker/                       # Docker build specifications
│   ├── api.Dockerfile            # Backend container with preloaded weights
│   ├── web.Dockerfile            # Nginx web server container
│   └── cli.Dockerfile            # Standalone CLI container
├── tests/                        # Automated unit tests
│   ├── test_core.py              # Core preprocessing, marker, and parser tests
│   └── test_cancellation.py      # Task cancellation manager tests
├── docker-compose.yml            # Primary Docker Compose configuration (API + Web)
├── docker-compose.gpu.yml        # Compose overlay for NVIDIA GPU acceleration
└── pyproject.toml                # Project metadata, dependencies & tool configuration
```

---

## 🐳 Running with Docker (Recommended)

The easiest way to run the Web UI and API service is via Docker Compose:

```bash
docker compose up --build -d
```

Once started, open your browser and navigate to:
👉 **[http://localhost:8080](http://localhost:8080)**

### For Servers with NVIDIA GPUs:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

---

## 💻 Local Installation & CLI Usage

If you prefer running VeriText natively (especially to leverage **Apple Silicon GPU / Metal MPS** on macOS):

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Analyze Raw Text
```bash
veritext --text "In today's rapidly evolving digital landscape, artificial intelligence plays a crucial role..."
```

For machine-readable output:
```bash
veritext --text "Your text here" --json
```

### 3. Analyze a Document (PDF / DOCX / MD / TXT)
```bash
veritext --file essay.pdf
```

### 4. Batch Score a Tabular Dataset (CSV / TSV)
```bash
veritext --input dataset.csv --text-column text --output scored_dataset.csv
```

### 5. Run Native API Server
```bash
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Running Tests

Run the test suite using Python's built-in `unittest` runner or `pytest`:

```bash
python3 -m unittest discover -s tests -v
# or with pytest:
pytest
```

---

## ⚡ Performance & Hardware Acceleration

| Platform | Inference Mode | Backend | Typical Latency (500 words) |
| :--- | :--- | :--- | :--- |
| **Apple Silicon (M1–M4)** | Native Python | Metal (`mps`) | **~30–60 ms** |
| **Apple Silicon (M1–M4)** | Docker (OrbStack) | ARM64 CPU | **~80–140 ms** |
| **NVIDIA GPU** | Docker / Native | CUDA (`cuda:0`) | **~15–35 ms** |
| **x86_64 Server** | Docker / Native | CPU | **~120–220 ms** |

> **Note on Apple Silicon in Docker:** Docker containers on macOS run inside a Linux virtual machine. Apple's hypervisor does not provide Metal GPGPU passthrough to guest Linux kernels. For maximum GPU acceleration on Mac hardware, run the application natively via Python virtual environment where PyTorch directly uses `mps`.

---

## 🔬 Acknowledgments & Attribution

This project originated as a fork of [**fast-ai-detector**](https://github.com/Ejhfast/fast-ai-detector) by Ethan Fast (`@Ejhfast`). 

While the initial codebase explored a 40M-parameter distilled Gemma model with Sparse Autoencoder (SAE) contrast vectors trained on the RAID dataset, **VeriText has been substantially re-architected and transformed into a production-grade multi-lingual detector**:

1. **Model Stack Evolution:** Replaced the distilled Gemma/RAID contrast pipeline with dedicated state-of-the-art transformer encoders for Russian (`toiletsandpaper/russian-ai-text-detector-bert` trained on the `LLMTrace` benchmark) and English (`mdrakibali/deberta-ai-detector-v3` based on `microsoft/deberta-v3-large`).
2. **Canonical Preprocessing Pipeline:** Designed and implemented a 21-rule text normalization engine that resolves false-positive edge cases stemming from Slavic typography (`ё`, `«»`, `—`, `…`) and Markdown markup.
3. **Adaptive Sliding Window:** Introduced dynamic stride-window chunking to reliably score documents far exceeding standard 512-token limits.
4. **Document & Dataset Parsers:** Added native extractors for `.pdf`, `.docx`, `.md`, and tabular `.csv`/`.tsv` scoring with word-weighted calibration.
5. **Full-Stack Architecture:** Developed an isolated two-tier service with a modern FastAPI backend, Nginx reverse proxy, and interactive Web UI with paragraph heatmaps.

We express our gratitude to **Ethan Fast** for the foundational inspiration and to the open-source community behind Hugging Face, RuBERT, DeBERTa, and LLMTrace.

### 🤖 AI-Assisted Engineering

The comprehensive architectural overhaul, algorithmic refinements (normalization rules, sliding-window chunking, document parsers), multilingual transformer integrations, test suites, and Docker deployment infrastructure in this project were designed and implemented with the assistance of **Artificial Intelligence** (AI-assisted engineering / pair programming).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
