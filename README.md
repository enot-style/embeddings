# Embeddings

**Embeddings** is a lightweight FastAPI microservice that mirrors the **OpenAI embeddings endpoint** (`/v1/embeddings`) while running **local Hugging Face text models**. It focuses on **text → embedding (vector)** and keeps everything else out of scope.

The service is intentionally minimal: you supply text (or token IDs) and a model ID, and it returns embedding vectors.

## Use cases

Embeddings is a good fit if you need:

* a **dedicated text embedding service** for similarity search, clustering, or indexing
* clean integration with **other backend systems**
* a **stable, minimal API** you control

Common use cases include:
- semantic search
- nearest-neighbor retrieval
- clustering and grouping of related documents
- duplicate or near-duplicate detection
- anomaly or outlier detection

## What Embeddings does

For each input text, Embeddings returns a **single embedding vector**:

* generated via **mean pooling** over the model’s token embeddings
* **L2-normalized**

## What Embeddings does *not* do

* No similarity or “compare” endpoints
* No task-specific heads (classification, re-ranking, etc.)
* No fine-tuning or training
* No model-specific pooling logic

Embeddings is a **pure text embedding microservice**.

## OpenAI API compatibility

The API is designed to mirror OpenAI’s embeddings endpoint as closely as possible:

### Endpoint

```
POST /v1/embeddings
```

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

> ⚠️ **Security & production readiness notice**
>
> Embeddings is provided as a **development tool** and is **not production-ready** by default:
> - No authentication or authorization
> - API docs (`/docs`, `/redoc`, `/openapi.json`) are public
> - No rate limiting or abuse protection
> - No security hardening or penetration testing

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
