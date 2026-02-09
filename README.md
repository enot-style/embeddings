# Embeddings

A lightweight, self-hosted embeddings microservice that mirrors the OpenAI **`POST /v1/embeddings`** API — but runs **local Hugging Face** text embedding models.

Use it as a drop-in embeddings backend for:
- semantic search & nearest-neighbor retrieval
- clustering / grouping
- duplicate detection
- offline indexing pipelines

**Scope:** text → vector only. No reranking, no training, no “compare” endpoints — just fast, predictable embeddings behind a stable API.

## Features

- OpenAI-compatible endpoint: `POST /v1/embeddings`
- Local inference with Hugging Face models (allowlist via `supported_models.txt`)
- Input formats: string(s) or token id(s)
- Output formats: float arrays or base64 (`encoding_format`)
- Optional dimension truncation (`dimensions`)
- CPU and GPU Docker builds + docker-compose
- Deterministic pooling: mean pooling + L2 normalization

## Quickstart

Build:

```bash
git clone https://github.com/kotylevskiy/embeddings.git
cd embeddings
docker compose up -d --build
```

Test:

```bash
curl -s http://localhost:11445/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"jinaai/jina-embeddings-v2-base-en","input":"Hi! This is a test"}'
```

The first call can take a while because the service needs to download the model from Hugging Face.

## OpenAI API compatibility

The API is designed to mirror OpenAI’s embeddings endpoint as closely as possible:

### Endpoint

```
POST /v1/embeddings
```

### API schema

`GET /schema.json` — OpenAPI schema

### Request fields

* `model` — required, must be in `supported_models.txt`
* `input` — required; one of:
  - a string
  - a list of strings
  - a list of integer token IDs
  - a list of token ID arrays
* `encoding_format` — optional: `"float"` (default) or `"base64"`
* `dimensions` — optional: if set, embeddings are **truncated** to this size
* `user` — optional: accepted and ignored (kept for spec parity)

### Response fields

* `object`: `"list"`
* `data`: list of `{ object, index, embedding }`
* `model`: echoes the request model
* `usage`: `{ prompt_tokens, total_tokens }`

Token usage is computed using the **Hugging Face tokenizer** for the chosen model.

### Compatibility notes

* `dimensions` performs a **simple truncation** (no re-projection)
* If `encoding_format="base64"`, embeddings are encoded as **base64-encoded float32**
* Token ID inputs must match the model’s tokenizer vocabulary

## Supported models

Embeddings only loads models listed in `supported_models.txt`.

Add new models by appending their Hugging Face IDs to that file.

## Embedding semantics

The service uses a **generic mean pooling** strategy:

```
embedding = mean(last_hidden_state * attention_mask)
```

This produces a single vector per input. Normalization is always enabled.

## Requirements & Deployment

### Before Publishing

Make sure you have, at minimum:
- an API gateway or reverse proxy with TLS termination
- an external auth/authorization service
- rate limiting and abuse protection
- basic monitoring (logs + metrics + alerts)

### Authentication (optional)

Set `EMBEDDINGS_API_KEYS` to a comma/space-separated list of API keys to require
`Authorization: Bearer <key>` on `POST /v1/embeddings`. Leave it empty to disable auth.

### Hugging Face access token

For public models, `HF_TOKEN` is optional. For gated/private models, you must set:

```bash
export HF_TOKEN=your_token
```

The token is used **only to download model weights**.

### CPU vs GPU

Embeddings runs on CPU-only systems, but larger models can be slow. For high throughput or large models, GPU is recommended.

## Running with Docker

### CPU ONLY processing via Docker Compose

Create a `.env` file with required environment variables (at minimum `EMBEDDINGS_PORT` if you want to change it):

```dotenv
HF_TOKEN=your_token_if_needed
EMBEDDINGS_PORT=11445
```

Then run:

```bash
git clone https://github.com/your-org/embeddings.git
cd embeddings
docker compose up -d
```

### GPU processing via Docker Compose

> ⚠️ **Docker GPU passthrough via NVIDIA Container Toolkit is supported only on Linux hosts.**

Set `PYTORCH_TAG` in your `.env` file to use `Dockerfile.gpu`.
Choose a tag with a CUDA version supported by your NVIDIA driver.

Run the GPU stack with:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
```

### Optional quantization (bitsandbytes, any model)

If you want lower VRAM usage, you can enable **bitsandbytes** quantization for any model:

```dotenv
EMBEDDINGS_BITSANDBYTES=8bit # or 4bit
```

Notes:
- bitsandbytes only works on CUDA.
- `bitsandbytes` must be installed (it’s included in `requirements.txt`).


### Jina v3 performance notes (flash-attn)

`jinaai/jina-embeddings-v3` can use **flash-attn** for faster attention on CUDA.
If flash-attn is not installed, you may see:

```
flash_attn is not installed. Using PyTorch native attention implementation.
```

To enable flash-attn in the GPU image, set this in `.env` before building:

```dotenv
INSTALL_FLASH_ATTN=1
```

Rebuild the GPU image after changing it.

> [!WARNING]
> Building flash-attn is time-consuming and highly resource-intensive.


## Example requests

### Single input

```bash
curl -sS http://localhost:11445/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

### Batch + base64 encoding

```bash
curl -sS http://localhost:11445/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "encoding_format": "base64",
    "input": [
      "First text",
      "Second text"
    ]
  }'
```

## Tests

A few curl-based tests live in `tests/`. Run them against a running container:

```bash
./tests/test_simple.sh
./tests/test_base64.sh
./tests/test_batch.sh
```
