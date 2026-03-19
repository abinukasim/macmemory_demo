# MacMemory Demo

## Overview

MacMemory is a local semantic search system for a single folder on macOS. Instead of relying on filenames or keyword matching, it indexes files into a shared embedding space so you can search by meaning across:

- text files
- PDFs
- images

The intended product experience is a Raycast command backed by a local Python API and a local ChromaDB vector store.

## What Is Implemented

The current repo now includes:

- a FastAPI backend with `GET /health` and `POST /search`
- a rebuild-style indexing CLI for `data/input`
- Gemini embedding integration for text queries, native small-PDF embeddings, text fallback for larger PDFs, and images
- local Chroma persistence
- text and PDF chunking with metadata
- thumbnail generation for image files
- file-level search results aggregated from chunk-level matches
- a Raycast extension manifest and search command client
- automated tests for helpers, API behavior, aggregation, and end-to-end local indexing with a fake embedder

## Architecture

`data/input` -> indexer -> Gemini embeddings -> ChromaDB -> FastAPI -> Raycast

### Backend flow

1. Scan `data/input`
2. Detect supported files
3. Load text, process a PDF, or process an image
4. Use native PDF embeddings for small PDFs and text chunking as a fallback for larger PDFs
5. Generate embeddings
6. Persist vectors + metadata in Chroma
7. Accept query text through `/search`
8. Query Chroma and aggregate results to the best hit per file

### Retrieval model

- text is indexed in chunks
- PDFs use native direct embeddings when supported and within the page limit, otherwise they fall back to text chunking
- images are indexed one record per file
- search returns file-level results
- the best matching chunk/page supplies the preview text

## Supported File Types

- text: `.txt`, `.md`
- images: `.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`
- documents: `.pdf`

## Repo Layout

```text
macmemory_demo/
├── app/
│   ├── api/                # FastAPI routes
│   ├── cli/                # Indexing and demo-data CLIs
│   ├── core/               # Settings and constants
│   ├── embeddings/         # Gemini adapter
│   ├── indexing/           # Chunking, loaders, indexing pipeline
│   ├── models/             # Pydantic schemas
│   ├── services/           # Search orchestration
│   ├── storage/            # Chroma wrapper
│   ├── utils/              # File, image, and demo helpers
│   └── main.py             # FastAPI app
├── data/
│   ├── chroma/             # Local vector DB files
│   ├── input/              # Folder to index
│   └── thumbs/             # Generated thumbnails
├── raycast/                # Raycast extension
├── tests/                  # Python test suite
├── .env.example
├── requirements.txt
└── README.md
```

## Key Commands

### Create a demo corpus

This generates a small text/PDF/image dataset in `data/input`:

```bash
./.venv/bin/python -m app.cli.seed_demo_data
```

### Index the folder

Full rebuild is the default:

```bash
./.venv/bin/python -m app.cli.index_folder
```

Optional flags:

- `--input-dir PATH`
- `--rebuild/--no-rebuild`

### Start the local API

```bash
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run tests

```bash
./.venv/bin/python -m pytest tests
./.venv/bin/python -m compileall app tests
```

## Setup

### 1. Create and activate a virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install OCR support for text-heavy images

```bash
brew install tesseract
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Required:

- `GEMINI_API_KEY`
- `GEMINI_EMBEDDING_MODEL`

Optional:

- `INPUT_DIR`
- `CHROMA_DIR`
- `THUMBS_DIR`
- `CHROMA_COLLECTION`
- `API_HOST`
- `API_PORT`

## Recommended Demo Flow

1. Set `GEMINI_API_KEY` and `GEMINI_EMBEDDING_MODEL` in `.env`
2. Install OCR support with `brew install tesseract` if you want screenshots and text-heavy images to contribute OCR text
3. Seed demo files with `python -m app.cli.seed_demo_data`
4. Index them with `python -m app.cli.index_folder`
5. Start the API with `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
6. Start the Raycast extension from `raycast/`
7. Search for phrases like:
   - `semantic retrieval`
   - `budget finance`
   - `sunset beach`

## Raycast

The `raycast/` directory contains a working extension manifest and a search command that:

- debounces input
- calls `POST /search`
- renders modality and score metadata
- shows previews and image thumbnails when available
- opens files and reveals them in Finder

To work on the extension:

```bash
cd raycast
npm install
npm run dev
```

If you want `npm run lint` to pass, set the `author` field in `raycast/package.json` to your real Raycast username.

## Verification

Verified locally on the Python side with:

```bash
./.venv/bin/python -m pytest tests
./.venv/bin/python -m compileall app tests
```

Current automated coverage includes:

- supported file-type detection
- stable file IDs
- chunking behavior
- search result aggregation
- `/health` and `/search` behavior
- end-to-end local indexing/search with temporary Chroma storage and a fake embedder

## Important Notes

- This is an interview MVP, not a production system.
- Indexing is rebuild-only for now. There is no watcher or incremental sync.
- Search is pure vector similarity with no reranking.
- Only one local folder is indexed.
- PDF indexing now prefers Gemini Embedding 2's native direct-PDF path for small PDFs and falls back to extracted-text chunking when native PDF embeddings are unavailable or the document is larger than the supported limit.
- `.webp` and `.heic` inputs are normalized to Gemini-supported PNG/JPEG bytes before image embedding.
- Image embeddings still depend on the configured Gemini model/provider supporting multimodal embedding requests. If that support is unavailable, image indexing will fail cleanly and the rest of the index can still proceed.
- Raycast build passes locally, but Raycast lint still requires a valid publishable `author` slug in `raycast/package.json`.

## Key Design Decision

Raycast is used instead of a web app because it:

- fits directly into the macOS workflow
- makes the demo feel closer to Spotlight than a generic dashboard
- keeps the product experience aligned with local search behavior
