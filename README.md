# MacMemory Demo

## Overview

MacMemory is a local semantic search system for a single folder on macOS. Instead of relying on filenames or keyword matching, it indexes files into a shared embedding space so you can search by meaning across:

- text files
- PDFs
- images

The intended product experience is a Raycast command backed by a local Python API and a local ChromaDB vector store.

## Why This Project Matters

MacMemory is a concrete answer to a common product gap on macOS: local file search is still strongest at filenames and literal keywords, but much weaker at intent. This project explores what a more modern local memory layer could feel like when text, PDFs, screenshots, and photos can all be retrieved from the same natural-language query.

It is also a strong engineering demo because it combines multimodal indexing, local-first architecture, vector retrieval, API design, and a polished Raycast client in one focused system. The result is small enough to explain in an interview, but deep enough to show product judgment and systems thinking.

## What Is Implemented

The current repo now includes:

- a FastAPI backend with `GET /health` and `POST /search`
- a rebuild-style indexing CLI for `data/input`
- Gemini integration for text queries, structured image descriptions, native small-PDF embeddings, and text fallback for larger PDFs
- local Chroma persistence
- text and PDF chunking with metadata
- thumbnail generation for image files
- OCR extraction for text-heavy images when `tesseract` is installed
- multi-vector image indexing using description, tags, concepts, and OCR text
- file-level search results aggregated from chunk-level matches
- search-time reranking with modality balancing, field-aware boosts, and folder-context signals
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
- images are indexed with multiple records per file
- image search uses structured captions, tags, concepts, optional OCR text, and local reranking
- search returns file-level results
- the best matching chunk/page supplies the preview text

## Technical Highlights

- Local-first architecture: all indexing, vector storage, search, and previews run on-device
- Cross-modal retrieval: a single query path can surface text, PDFs, screenshots, and photos
- Structured image indexing: images are enriched into caption, tags, concepts, and OCR-aware retrieval text instead of relying only on filenames
- Multi-vector strategy: each image can contribute several indexed records, which improves recall while preserving file-level results
- Search ranking beyond raw vector distance: the backend blends embedding similarity with folder context, OCR text, tags, concepts, and modality balancing
- Practical macOS integration: Raycast makes the demo feel like a real productivity tool rather than a generic web dashboard

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

## Demo in 30 Seconds

1. Put a few PDFs, notes, screenshots, and photos into `data/input`
2. Run `./.venv/bin/python -m app.cli.index_folder --rebuild`
3. Start the API with `./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
4. Start the Raycast extension from `raycast/`
5. Search for a natural phrase like `christmas tree`, `resume`, or `tracker screenshot`

The best version of the demo is when the query does not exactly match the filename, but still retrieves the right file because the system indexed meaning, OCR text, image descriptions, or folder context.

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

## Example Queries

- `christmas tree`
- `holiday market`
- `fireplace`
- `vienna christmas market`
- `resume`
- `tracker screenshot`
- `stress assessment`
- `sunset beach`
- `mountain lake`
- `budget planning`

These work well because they exercise different retrieval paths:

- direct semantic text retrieval
- PDF chunk retrieval
- OCR-driven screenshot retrieval
- image caption and concept retrieval
- folder-context-aware ranking

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

## Design Tradeoffs

- Rebuild-only indexing over incremental sync: simpler, more predictable, and easier to demo under time pressure
- Local Chroma persistence over managed infrastructure: lower operational overhead and better for privacy-oriented positioning
- Raycast over a custom frontend: faster path to a usable macOS-native interaction model
- Heuristic reranking over a separate learned reranker: easier to inspect, tune, and explain in an interview
- OCR as an optional local dependency: keeps the base install small while still supporting screenshots and text-heavy images when needed
- Multi-vector image indexing over a single caption vector: more storage and indexing work, but meaningfully better recall for images

## Important Notes

- This is an interview MVP, not a production system.
- Indexing is rebuild-only for now. There is no watcher or incremental sync.
- Search combines vector retrieval with local reranking and modality balancing.
- Only one local folder is indexed.
- PDF indexing now prefers Gemini Embedding 2's native direct-PDF path for small PDFs and falls back to extracted-text chunking when native PDF embeddings are unavailable or the document is larger than the supported limit.
- `.webp` and `.heic` inputs are normalized to Gemini-supported PNG/JPEG bytes before image understanding.
- OCR enrichment depends on a local `tesseract` install.
- Raycast build passes locally, but Raycast lint still requires a valid publishable `author` slug in `raycast/package.json`.

## Key Design Decision

Raycast is used instead of a web app because it:

- fits directly into the macOS workflow
- makes the demo feel closer to Spotlight than a generic dashboard
- keeps the product experience aligned with local search behavior

## Built For

- Demoing multimodal semantic search in a way that feels immediately understandable
- Technical interviews where architecture, tradeoffs, and implementation depth matter
- Resume and portfolio review, where one project should show both product sense and engineering range
- A local-first use case where privacy, responsiveness, and workflow fit are more important than cloud scale

## Future Work

- Incremental indexing and file change tracking instead of rebuild-only refreshes
- Better query-time explanations showing which field matched: caption, OCR, tags, concepts, or folder context
- Smarter OCR detection so only text-heavy images pay the OCR cost
- Optional user tagging or lightweight feedback signals to improve retrieval over time
- More adaptive ranking that changes modality balance based on query type
- A more deliberate corpus-management story for real personal archives beyond the demo folder

## Resume Summary

MacMemory is a local-first multimodal search prototype for macOS that retrieves text files, PDFs, screenshots, and photos by semantic meaning instead of keyword matching. It combines a Python indexing and search backend, Gemini-powered enrichment, Chroma vector storage, OCR support, multi-vector image indexing, and a Raycast client into a polished interview-ready system.
