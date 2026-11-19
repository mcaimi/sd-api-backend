# SD API Backend

A FastAPI-based backend service for Stable Diffusion image generation, inpainting, multi-model merging, and model management. Supports SD 1.5, SDXL, Illustrious XL, Flux, and Z-Image (Base and Turbo) pipelines with asynchronous job processing, model caching, and optional LoRA adapter integration.

This is a rewrite effort of the [SD-WebApp](https://github.com/mcaimi/sd-webapp) project, and has been brewing for a bit now.
The idea is to decouple the UI from the actual inference backend. This repo will host the backend code.

The Stable Diffusion pipelines, device selection, checkpoint merging and telemetry live in the [metastable](https://github.com/mcaimi/metastable) library ([PyPI](https://pypi.org/project/metastable/)); this repo is the REST API and web UI on top of it.

## Implemented Features

- **Text-to-Image Generation** - Submit async jobs for SD 1.5, SDXL, Illustrious XL, Flux, and Z-Image (Base and Turbo) models, receive results via job status polling
- **Inpainting** - Image inpainting using specialized pipelines with base image and mask (base64) inputs, including Flux's and Z-Image's strength-based re-noising inpaint
- **Batch Generation** - Submit multiple generation jobs in a single request
- **Model Comparison** - Generate images from multiple checkpoints side-by-side for comparison
- **Multi-Model Merging** - Merge two or more models using configurable methods (linear, slerp, additive, subtract), with weight coefficients and multi-step recipe execution for every architecture (Flux and Z-Image merge the transformer only). Merges are streamed with constant memory and checked from the file headers before they are queued
- **Asynchronous Job Queue** - Non-blocking API with worker-based async pipeline execution via `asyncio.Queue`; worker count and result TTL are configurable, and finished jobs are cleaned up after the TTL
- **Model Caching** - LRU cache (configurable, default 2 pipelines) for expensive diffusion model loading. Pipelines are leased to one job at a time and never evicted while in use, so concurrent workers can't clobber each other's LoRAs/VAE
- **LoRA Support** - Load and fuse PEFT-based LoRA adapters into the base pipeline with configurable strength
- **Custom VAE** - Replace the default VAE with custom models loaded from disk
- **Multi-Device Auto-Detection** - Automatic fallback: MPS (Apple Silicon) > CUDA (NVIDIA) / ROCm (AMD) > CPU with dtype auto-selection (float16 for SD/SDXL/Illustrious XL, bfloat16 for Flux/Z-Image, float32 for CPU)
- **AMD ROCm Support** - Config-driven backend selection (`device.back$end: rocm`) with `gfx_override` for cards missing from the ROCm support matrix (e.g. RX 6600 XT), plus `rocm-smi` telemetry
- **Flux/Z-Image Memory Offload** - Flux and Z-Image pipelines use `enable_model_cpu_offload()` instead of a full device move to fit their larger footprint (Flux: ~12B parameter transformer + T5-XXL encoder; Z-Image: 6B transformer + Qwen3 encoder) on consumer hardware
- **Configurable Output** - Optional disk persistence of generated images and metadata to `outputs/images/` and `outputs/json/` as PNG and JSON
- **System Endpoints** - Health checks, system info (GPU, cache stats, queue stats), and configuration exposure
- **CORS Middleware** - Configurable CORS support, enabled by default with wildcard origins

## API Endpoints

All endpoints are prefixed with `/api/v1`.

### Generation

| Method | Route | Description |
| -------- | ------- | ------------- |
| POST | `/sd15/generate` | Submit SD 1.5 text-to-image generation job |
| POST | `/sd15/inpaint` | Submit SD 1.5 inpainting job |
| POST | `/sd15/generate/batch` | Submit multiple SD 1.5 generation jobs |
| POST | `/sd15/generate/compare` | Generate from multiple SD 1.5 models for comparison |
| POST | `/sdxl/generate` | Submit SDXL text-to-image generation job |
| POST | `/sdxl/inpaint` | Submit SDXL inpainting job |
| POST | `/sdxl/generate/batch` | Submit multiple SDXL generation jobs |
| POST | `/sdxl/generate/compare` | Generate from multiple SDXL models for comparison |
| POST | `/illustrious/generate` | Submit Illustrious XL text-to-image generation job |
| POST | `/illustrious/inpaint` | Submit Illustrious XL inpainting job |
| POST | `/illustrious/generate/batch` | Submit multiple Illustrious XL generation jobs |
| POST | `/illustrious/generate/compare` | Generate from multiple Illustrious XL models for comparison |
| POST | `/flux/generate` | Submit Flux text-to-image generation job |
| POST | `/flux/inpaint` | Submit Flux inpainting job (strength-based re-noising) |
| POST | `/flux/generate/batch` | Submit multiple Flux generation jobs |
| POST | `/flux/generate/compare` | Generate from multiple Flux models for comparison |
| POST | `/zimage/generate` | Submit Z-Image text-to-image generation job (`variant`: `turbo` or `base`) |
| POST | `/zimage/inpaint` | Submit Z-Image inpainting job (strength-based re-noising) |
| POST | `/zimage/generate/batch` | Submit multiple Z-Image generation jobs |
| POST | `/zimage/generate/compare` | Generate from multiple Z-Image models (of one variant) for comparison |

Flux and Z-Image use the flow-matching scheduler set (`FlowMatch Euler`); the SD schedulers are rejected for them and vice versa. `FlowMatch Heun` is no longer available.

**Z-Image variants.** Base and Turbo checkpoints can't be told apart from their files, so Z-Image requests carry `variant` (default `turbo`). It selects the matching components directory and the defaults of omitted settings:

| variant | `steps` | `cfg_scale` | negative prompt |
|---------|---------|-------------|-----------------|
| `turbo` | 8 | 0.0 | ignored (CFG off) |
| `base`  | 28 | 4.0 | used |

CFG is enabled whenever `cfg_scale > 0`. Downloaded Civitai checkpoints show their base model (`ZImageTurbo`/`ZImageBase`) in the models listing.

### Model Management

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/models/{model_type}/{resource_type}` | List available models (`model_type`: sd15/sdxl/illustrious/flux/zimage, `resource_type`: checkpoints/loras/vaes) |
| GET | `/models/{model_type}/{resource_type}/{model_name}/metadata` | Get safetensors metadata for a specific model |

### Model Merging

| Method | Route | Description |
| -------- | ------- | ------------- |
| POST | `/merge/merge` | Merge two models into one new checkpoint with configurable method (linear, slerp, additive, subtract), alpha weight, and metadata preservation |
| POST | `/merge/batch` | Merge one base model against multiple target models simultaneously, saving each output into a specified subdirectory |
| POST | `/merge/recipe` | Execute a multi-step merge recipe where each step takes the current result and merges it with a new target model using its own method/alpha |

> `alpha` is the weight of the target model for every method (0.0 = base, 1.0 = target). **Breaking change:** before the move to metastable, `linear` used the opposite convention; requests and recipes written for it need `alpha → 1 - alpha`.

> Merging supports `sd15`, `sdxl`, `illustrious`, `flux`, and `zimage`. Every merge request takes an optional `output_dtype` (`auto`, `bfloat16`, `float16`, `float32`; `auto` keeps the base model's dtypes).
>
> Before a merge is queued, all checkpoints are checked from their safetensors headers (keys, shapes, key layout, no fp8 weights); incompatible checkpoints are rejected with HTTP 400. For batch merges a single incompatible target rejects the whole request.
>
> **Flux and Z-Image** merge only the diffusion transformer. Both inputs must use the same key layout (original/ComfyUI or diffusers). In all-in-one files the VAE and text encoders are copied unchanged from the base. fp8 checkpoints are rejected: use a bf16/fp16/fp32 version. Z-Image Base and Turbo files can't be told apart, so nothing stops merging them, but the result mixes a distilled and a non-distilled model and its right steps/CFG become unpredictable. Flux dev and schnell checkpoints have different keys and are rejected.

### Job Management

| Method | Route | Description |
| -------- | ------- | ------------- |
| GET | `/jobs` | List all jobs with current status |
| GET | `/jobs/{job_id}` | Get detailed status and result for a specific job |
| DELETE | `/jobs/{job_id}` | Cancel a pending or running job |

### System & Info

| Method | Route | Description |
| -------- | ------- | ------------- |
| GET | `/system/health` | Health check endpoint |
| GET | `/system/info` | System info (GPU, data type, cache stats, queue stats) |
| GET | `/info/schedulers` | List available noise schedulers (`?model_type=` for an architecture's set; default: the SD set) |
| GET | `/info/app_settings` | Return current application configuration |

Additional Swagger UI docs at `/docs` and ReDoc at `/redoc` (auto-generated by FastAPI).

## Class Diagrams

### Core Pipeline Architecture

Provided by `metastable.pipelines`; generators are created with `create_generator(model_type, checkpoint, **kwargs)`.

```
┌─────────────────────────────────┐
│     BasePipelineGenerator       │  (ABC)
│  ─────────────────────────────  │
│ - model_checkpoint: str         │
│ - torch_device: str             │
│ - dtype: torch.dtype            │
│  ─────────────────────────────  │
│ + load_pipeline()               │
│ + load_inpaint_pipeline()       │
│ + to_device()                   │
│ + forward(prompt, params)       │  # text-to-image
│ + forward_inpaint(base, mask…)  │  # inpainting
│ + add_loras([LoraSpec])         │
│ + load_custom_vae()             │
│ + get_resolutions()             │
│ + SCHEDULERS (overridable)      │
└──────────┬──────────────────────┘
           │ inherits
    ┌───────┬───────┬───────┐
    │       │        │       │
┌────────┐┌────────┐┌─────────────┐┌────────┐
│ SD15   ││ SDXL   ││ Illustrious ││ TransformerPipeline-   │
│Pipeline││Pipeline││ Pipeline    ││ Generator              │
│Generator││Generator││ Generator  ││ (shared components dir,│
└────────┘└────────┘└─────────────┘│  bf16, cpu offload,    │
                                   │  per-dir lock)         │
                                   └──────────┬─────────────┘
                                       ┌──────┴──────┐
                                   ┌────────┐  ┌──────────┐
                                   │ Flux   │  │ Z-Image  │
                                   │Pipeline│  │ Pipeline │
                                   │Generator│ │ Generator│
                                   └────────┘  └──────────┘
                                   forward/forward_inpaint
                                   (true_cfg_scale, strength)
```

### Service Layer

```
┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  GenerationService   │────>│   ModelCacheManager  │────>│ BasePipelineGen  │
│ ──────────────────── │     │ ───────────────────  │     │ ───────────────  │
│ + generate()         │     │ + acquire()          │     │ - cache: dict    │
│ + inpaint()          │     │ + _load_pipeline()   │     │ + forward()      │
│ + batch_generate()   │     │ + clear_cache()      │     │ + add_loras()    │
│ + compare_models()   │     │ + get_stats()        │     └──────────────────┘
└──────────┬───────────┘     └─────────────────────┘
           │
           │ submits
           v
┌──────────────────────┐
│     JobQueue          │
│ ──────────────────── │
│ + submit_job()       │  # returns job_id
│ + get_job_status()   │
│ + cancel_job()       │
│ + list_jobs()        │
│ + stop_workers()     │
│ - _queue: Queue      │
│ - _jobs: dict        │
│ - _worker()          │  # async background loop
└──────────────────────┘
```

### Request / Response Models

```
┌─────────────────┐   ┌────────────────┐   ┌──────────────────┐
│ Generation      │   │ InpaintRequest │   │ LoraConfig       │
│ Request         │   │ ────────────── │   │ ──────────────── │
│ ──────────────  │   │ - image_base64 │   │ - name: str      │
│ - prompt: str   │   │ - mask_base64  │   │ - strength: float│
│ - model: str    │   │ - image: str   │   └────────┬─────────┘
│ - params: dict  │   │ - mask: str    │            │
│ - loras: list[] │──>│ - resolution   │            │
│ - vae: str|None │   │ - negative_prompt
│ ──────────────  │   └────────────────┘
│ + to_dict()     │
└────────┬────────┘
         │ validates
         v
┌─────────────────────────────────────────┐
│        JobResponse / JobStatusResponse   │
│  ─────────────────────────────────────  │
│  - job_id: str (UUID4)                  │
│  - status: Job (QUEUED/RUNNING/...)     │
│  - result: JobResult|None               │
│    - image_base64: str                  │
│    - metadata: dict                     │
│    - seed: int                          │
└─────────────────────────────────────────┘
```

### Configuration

```
┌──────────────────┐     ┌─────────────────┐
│ parameters.yaml  │────>│   AppConfig     │
│ ─────────────── │     │ ─────────────── │
│ checkpoints:     │     │ - _params: dict │
│   sd15: path     │     │ ─────────────── │
│   sdxl: path     │     │ + get_model_... │
│ loras:           │     │ + setup_paths() │
│ vaes:             │     │ + as_dict()     │
│ device:          │     └─────────────────┘
│ storage:         │
│ api:             │
│   queue:         │          ▲
│   cache:         │          │ lru_cache
│   cors:          │     ┌─────────────────┐
└──────────────────┘     │ get_app_config()│
                         └─────────────────┘
```

## Overall Call Flow

### Image Generation (Primary Workflow)

1. **Client** sends `POST /api/v1/{sd15,sdxl,illustrious,flux,zimage}/generate` with prompt, model, and generation parameters
2. **Router** (one per architecture, built by `create_generation_router()`) resolves model path and, for Flux/Z-Image, the shared components directory from `AppConfig`, converts LoRA configs to dicts, validates request
3. **JobQueue** generates a UUID4 `job_id`, creates a `Job` object, enqueues it, and returns immediately with `JobResponse(status="queued")`
4. **Background workers** (`_worker` loop) dequeue jobs and set status to `RUNNING`
5. **GenerationService** calls `ModelCacheManager.get_pipeline()` to get or load the diffusion pipeline
6. Pipeline loads in a **thread pool executor** (blocking diffusers call)
7. LoRA adapters are fused via `peft` and custom VAE is swapped if provided
8. Pipeline's `forward()` (text-to-image) or `forward_inpaint()` is invoked with generation params
9. Result is an **image** (base64-encoded PNG) plus **metadata** dict (`GenerationMetadata` container)
10. If `persist_to_backend` is enabled, image is saved to `outputs/images/{uuid}.png` and metadata to `outputs/json/{uuid}.json`
11. Job is marked `COMPLETED` and result stored for client polling

### Model Merging Call Flow

1. **Client** sends `POST /api/v1/merge/merge` (or `/merge/batch`, `/merge/recipe`) with model specs, merge method, and weight parameters
2. **Router** validates model paths from `AppConfig` and checks from the safetensors headers that the checkpoints can be merged
3. **JobQueue** generates a `job_id` and returns immediately with `JobResponse(status="queued")`
4. **Background workers** dequeue jobs and invoke `MergeService.merge_models()`
5. **MergeService** runs `metastable.merge.MergePipeline` for the model type: tensors are streamed one at a time from memory-mapped inputs, merged (linear, slerp, additive, or subtract) and written to the output checkpoint, so memory use stays constant even for 12B-parameter Flux transformers. Provenance goes to a JSON sidecar next to the output
6. **Recipe merges** apply every step to each tensor in turn, feeding the result of the previous step as the base for the next
7. Job is marked `COMPLETED` with the path to the newly created merged model

### Job Status Polling

```
Client: GET /api/v1/jobs/{job_id}
    → JobQueue.get_job_status()
    → Returns JobStatusResponse { job_id, job_type, status, result, error, timestamps }
    → If status=CALLED: extract result.image_base64 → decode → save PNG
    → If status=FAILED:  display result.error
```

### Model Cache Flow

```
async with ModelCacheManager.acquire(model_type, model_path, pipeline_type, state, components_dir)
  1. Build cache key: "{model_type}:{resolved model path}:{pipeline_type}[:{components dir}]"
  2. Take the pipeline's lock (pipelines built from the same components
     directory share one lock, since they share text encoders/VAE)
     → the lease is exclusive until the block exits
  3. Cache HIT with a different LoRA/VAE state applied → drop it and reload
  4. Cache HIT → update access order → yield cached pipeline
  5. Cache MISS → if at capacity (>= max_models) evict the least-recently-used
     pipeline that isn't leased; if all are leased, wait for one to be released
  6. _load_pipeline() in thread pool
     → create_generator() → load_pipeline() / load_inpaint_pipeline() → to_device()
  7. Insert new pipeline into cache and yield it
```

The generation service applies LoRAs and the custom VAE and runs inference inside
the lease. Each cached pipeline remembers which LoRA/VAE state is applied: a request
for the same state reuses it, a different one reloads the pipeline (fused LoRAs
can't be removed in place).

### Startup / Shutdown Flow

```
Uvicorn starts → api.app:app (pre-created via create_app())

Lifespan startup:
  1. Load AppConfig from parameters.yaml
  2. metastable.configure_device(config.device) — before any torch.cuda usage
  3. Create FastAPI app with CORS middleware
  4. Mount routers (sd15, sdxl, illustrious, flux, zimage, models, merge, jobs, system, info, outputs)
  5. JobQueue singleton created → start_workers() spawns N async worker tasks
  6. Log "SD API Backend started"

Lifespan shutdown:
  1. JobQueue.stop_workers() → sets _running=False, drains queue, cancels workers
  2. Log "SD API Backend shutdown complete"
```

## Usage Examples

### Python (Async with httpx)

```python
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000/api/v1"

async def generate_image():
    async with httpx.AsyncClient() as client:
        # Submit generation job
        response = await client.post(
            f"{BASE_URL}/sd15/generate",
            json={
                "prompt": "a beautiful sunset over mountains",
                "model": "v2-1024px-pruned.safetensors",
                "params": {
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "seed": 42
                }
            }
        )
        job = response.json()
        job_id = job["job_id"]
        print(f"Job submitted: {job_id}")

        # Poll for completion
        while True:
            status = await client.get(f"{BASE_URL}/jobs/{job_id}")
            data = status.json()
            if data["status"] == "completed":
                result = data["result"]
                # Decode and save image
                import base64
                with open(f"output_{job_id}.png", "wb") as f:
                    f.write(base64.b64decode(result["image_base64"]))
                print(f"Image saved. Seed: {result['seed']}")
                break
            elif data["status"] == "failed":
                print(f"Job failed: {data['error']}")
                break
            await asyncio.sleep(1)

asyncio.run(generate_image())
```

### cURL

```bash
# Submit a generation job
curl -X POST http://localhost:8000/api/v1/sd15/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cyberpunk city at night",
    "model": "v2-1024px-pruned.safetensors",
    "params": {
      "width": 512,
      "height": 512,
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "seed": 42
    }
  }'

# Submit a Flux generation job
curl -X POST http://localhost:8000/api/v1/flux/generate \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "a photo of an astronaut riding a horse on mars",
    "model_checkpoint": "flux1-dev.safetensors",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "cfg_scale": 3.5,
    "scheduler": "FlowMatch Euler",
    "seed": 42
  }'

# Check job status
curl http://localhost:8000/api/v1/jobs/{job_id}

# List all jobs
curl http://localhost:8000/api/v1/jobs

# Cancel a job
curl -X DELETE http://localhost:8000/api/v1/jobs/{job_id}

# List available checkpoints
curl http://localhost:8000/api/v1/models/sd15/checkpoints

# Submit a Z-Image Turbo generation job (8 steps, CFG off by default)
curl -X POST http://localhost:8000/api/v1/zimage/generate \
  -H "Content-Type: application/json" \
  -d '{
    "positive_prompt": "a red fox in fresh snow, golden hour",
    "model_checkpoint": "z_image_turbo_bf16.safetensors",
    "variant": "turbo",
    "seed": 42
  }'

# Merge two models
curl -X POST http://localhost:8000/api/v1/merge/merge \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "flux",
    "base_model": "flux1-dev.safetensors",
    "target_model": "flux1-dev-finetune.safetensors",
    "method": "linear",
    "alpha": 0.5,
    "output_name": "flux1-dev-merged",
    "output_dtype": "bfloat16"
  }'

# Multi-step recipe merge
curl -X POST http://localhost:8000/api/v1/merge/recipe \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sdxl",
    "base_model": "base.safetensors",
    "steps": [
      {"target_model": "style1.safetensors", "method": "additive", "alpha": 0.4},
      {"target_model": "style2.safetensors", "method": "linear", "alpha": 0.3}
    ],
    "output_name": "recipe_merged"
  }'

# Health check
curl http://localhost:8000/api/v1/system/health
```

### With LoRA Adapters

```python
response = await client.post(
    f"{BASE_URL}/sdxl/generate",
    json={
        "prompt": "a portrait in anime style",
        "model": "sd_xl_base_1.0.safetensors",
        "params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 25,
        },
        "loras": [
            {"name": "anime_style.safetensors", "strength": 0.8}
        ]
    }
)
```

## Configuration

Copy `parameters.yaml.example` to `parameters.yaml` and adjust paths:

```yaml
checkpoints:
  sd15:
    path: 'models/checkpoints/SD15'
  sdxl:
    path: 'models/checkpoints/SDXL'
  illustrious:
    path: 'models/checkpoints/Illustrious'
  flux:
    path: 'models/checkpoints/Flux'
  zimage:
    path: 'models/checkpoints/ZImage'
loras:
  sd15:
    path: 'models/loras/SD15'
  sdxl:
    path: 'models/loras/SDXL'
  illustrious:
    path: 'models/loras/Illustrious'
  flux:
    path: 'models/loras/Flux'
  zimage:
    path: 'models/loras/ZImage'
vaes:
  sd15:
    path: 'models/vae/SD15'
  sdxl:
    path: 'models/vae/SDXL'
  illustrious:
    path: 'models/vae/Illustrious'
  flux:
    path: 'models/vae/Flux'
  zimage:
    path: 'models/vae/ZImage'
# Flux checkpoints are transformer-only single-file checkpoints; the shared
# text encoders (CLIP-L + T5-XXL), tokenizers, VAE, and scheduler are loaded
# once from a local diffusers-format directory and reused across checkpoints.
flux:
  components_path: 'models/flux_components'
# Z-Image Turbo and Base each need the components directory of their own
# repository (the scheduler shift differs); requests pick one with "variant".
zimage:
  components:
    turbo: 'models/zimage_turbo_components'
    base: 'models/zimage_components'
storage:
  persist_to_backend: true
  output_images: 'outputs/images'
  output_json: 'outputs/json'
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  job_queue:
    num_workers: 1
    result_ttl: 3600
  model_cache:
    max_models: 2
  cors:
    enabled: true
    allow_origins: ["*"]
```

#### Concurrent GPU jobs

`api.job_queue.num_workers` sets how many generation/merge jobs run at once (default 1).
With more than one worker:

- jobs on **different** checkpoints run concurrently; jobs on the same checkpoint
  (and pipeline type) still run one after another;
- jobs sharing a **components directory** run one at a time, because those
  checkpoints share the same text encoders and VAE: all Flux jobs, all Z-Image
  Turbo jobs, all Z-Image Base jobs. Flux and Z-Image jobs (or Turbo and Base)
  can run concurrently;
- peak VRAM/RAM grows with every job running at once, and on a single GPU total
  throughput rarely improves — the main win is overlapping model loading or merges
  with generation. Keep `model_cache.max_models` >= `num_workers`, otherwise
  workers wait for a free cache slot.

### Flux Setup

Flux checkpoints only ship the transformer; the shared text encoders, tokenizers, VAE, and scheduler must be populated once under `flux.components_path` in standard diffusers subfolder layout (`text_encoder/`, `tokenizer/`, `text_encoder_2/`, `tokenizer_2/`, `vae/`, `scheduler/`):

```bash
# FLUX.1-schnell is ungated and license-friction-free for testing;
# FLUX.1-dev requires accepting its gated license on Hugging Face first.
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --exclude "transformer/*" --local-dir models/flux_components
```

Then place a single-file Flux transformer checkpoint (e.g. `flux1-dev.safetensors`) under `checkpoints.flux.path`.

If the components directory also contains `transformer/config.json`, it is used to load the checkpoint, so loading works offline (otherwise diffusers infers the config from the checkpoint and may fetch it from the Hugging Face Hub). Keep it by excluding only the weights: `--exclude "transformer/*.safetensors"`.

### Z-Image Setup

Z-Image works like Flux: checkpoints ship the transformer, and the Qwen3 text encoder, tokenizer, VAE, and scheduler come from a diffusers-format directory. Turbo and Base each need the directory from their own repository, configured under `zimage.components.turbo` and `zimage.components.base` (leave out a variant you don't use; requests for it are rejected with HTTP 400):

```bash
huggingface-cli download Tongyi-MAI/Z-Image-Turbo \
  --exclude "transformer/*.safetensors" \
  --local-dir models/zimage_turbo_components
huggingface-cli download Tongyi-MAI/Z-Image \
  --exclude "transformer/*.safetensors" \
  --local-dir models/zimage_components
```

Then place single-file Z-Image transformer checkpoints (e.g. `z_image_turbo_bf16.safetensors`) under `checkpoints.zimage.path`, and send `"variant": "base"` with requests for Base checkpoints.

A `parameters.yaml` written before Z-Image support keeps working: Z-Image endpoints answer HTTP 400 until the `zimage` sections are added.

### AMD ROCm Support (older cards, e.g. RX 6600)

The server runs on AMD GPUs through PyTorch's ROCm builds. Since PyTorch
implements HIP behind the CUDA API, transformers/torch report these devices as
"CUDA" — the server disambiguates them and labels the backend `rocm` in
`GET /api/v1/system/info` (`torch.version.hip` is set on ROCm builds).

1. **Install the ROCm torch build** — swap in the Linux variant of the project
   file, which carries the `rocm10.0` extra pointing at the official PyTorch
   ROCm wheel index:

   ```bash
   cp pyproject.linux.toml pyproject.toml
   uv sync --extra rocm10.0
   ```

2. **Select the backend in `parameters.yaml`** (optional section; defaults are
   auto-detect in order MPS > CUDA/ROCm > CPU):

   ```yaml
   device:
     backend: rocm          # force the AMD backend (auto also works once detectable)
     index: 0               # GPU ordinal on multi-GPU hosts
     dtype: auto            # auto | float16 | bfloat16 | float32
     # gfx_override: "10.3.0"
     # env: {}
   ```

3. **Cards missing from the ROCm support matrix** — many older GPUs are not
   officially supported but work when the HSA runtime is told to treat them as
   a supported variant. The RX 6600 (Navi 23, `gfx1032`) is the canonical
   example: set `gfx_override: "10.3.0"` and the runtime presents it as
   `gfx1030` (supported RDNA2 flagship family, e.g. Radeon PRO W6800). This
   must happen before HIP initialization, which the server handles itself —
   no manual environment setup needed. Common overrides:

   | Card | Actual IP | `gfx_override` |
   | ------ | ----------- | ---------------- |
   | RX 6700 XT / 6750 XT (Navi 22) | gfx1031 | `10.3.0` |
   | RX 6600 / 6600 XT / 6650 XT (Navi 23) | gfx1032 | `10.3.0` |
   | RX 6500 XT / 6400 (Navi 24) | gfx1034 | `10.3.0` |

   > Polaris/Vega cards (RX 4xx/5xx, gfx803) are not covered: ROCm dropped
   > them after 5.1.7 and no override revives them on current ROCm releases.
   > RDNA1 cards (RX 5700/5600 = Navi 10, `gfx1010`; RX 5500 XT/5500 =
   > Navi 14, `gfx1012`) are in the same position — no supported same-family
   > target exists in modern ROCm, so `10.3.0` does not apply to them either.

   Notes and expectations:
   - `dtype: auto` keeps **float16** on ROCm; bfloat16 is not reliably
     supported on older RDNA silicon. Set `dtype: bfloat16` explicitly on
     RDNA2+ cards if you want Flux's default bf16 path.
   - Telemetry: VRAM stats work out of the box (via the torch CUDA API);
     compute utilization uses a best-effort `rocm-smi` call and reports an
     honest "not available" when the CLI is missing.
   - It might be necessary to downgrade all ROCm packages to pre 6.X versions
     to fully support an older card. A container image is clearly preferred for
     this use case and one may come in the future.
   - A 4 GB variant card, if supported, is SD 1.5 territory only; also, 
     keep `api.model_cache.max_models: 1`.
   - SDXL/Illustrious need at least 8GB of VRAM. For Flux, stick to modern,
     supported ROCm cards

### Running

```bash
# Install dependencies
uv sync

# Start the server
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run the test suite
uv run pytest
```

### Container

The `Containerfile` builds a Debian-based image; dependencies are synced with `uv` and the server runs as an unprivileged user (OpenShift arbitrary-UID compatible, writable paths are group `0`). It listens on `0.0.0.0` by default.

```bash
podman build -t sd-api-backend -f Containerfile .

podman run -p 8000:8000 \
  -v ./models:/opt/app-root/data/models:Z \
  -v ./outputs:/opt/app-root/data/outputs:Z \
  sd-api-backend
```

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `SD_API_HOST` | `0.0.0.0` | Listening address |
| `SD_API_PORT` | `8000` | Listening port |
| `SD_API_MODELS_DIR` | `/opt/app-root/data/models` | Root for checkpoints, LoRAs, VAEs, and Flux components |
| `SD_API_OUTPUTS_DIR` | `/opt/app-root/data/outputs` | Root for persisted images and JSON metadata |
| `SD_API_PERSIST_OUTPUTS` | `true` | Persist generated outputs to disk |
| `SD_API_DEVICE_BACKEND` | `auto` | `auto` / `cuda` / `rocm` / `cpu` |
| `CONFIG_FILE` | `/opt/app-root/config/parameters.yaml` | Generated from the variables above at startup unless the file already exists (mount your own `parameters.yaml` here to take full control) |

The same variables are also build arguments (`--build-arg SD_API_PORT=...`). The default build uses `pyproject.toml` (PyPI torch, CUDA-enabled on Linux); a different project file and extras can be selected with `--build-arg PYPROJECT=pyproject.linux.toml --build-arg UV_SYNC_ARGS="--extra cu134"`.
