# Raycast Extension

This directory contains a working Raycast command that queries the local FastAPI backend.

Development flow:

- install dependencies with `npm install`
- start the backend on `http://127.0.0.1:8000`
- run `npm run dev` inside `raycast/`
- use the `Search Local Memory` command in Raycast

The command:

- debounces query text
- calls `POST /search`
- shows modality + score metadata
- opens files and reveals them in Finder
