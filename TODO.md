## TODO

Many features still need to be ported over from the [sd-webapp](https://codeberg.org/mcaimi/sd-webapp) project.

The groundwork for everything has been ported over (e.g. simply copied from the old repo :))

This document outlines what it is missing and what will *eventually* come:

### Model Merging Pipeline

- **merge features** - Mostly Done. Model merge APIs are exposed via `api/routers/merge.py` (`/merge/merge`, `/merge/batch`, `/merge/recipe`), backed by `api/services/merge.py`, the Pydantic models in `api/models/merge.py`, and the merge logic in the [metastable](https://github.com/mcaimi/metastable) library's `metastable.merge` package (supporting LINEAR, SLERP, ADDITIVE, SUBTRACT methods). Currently supports `sd15`, `sdxl`, and `illustrious` checkpoints; see the Flux section below for what's still missing.

### Flux Model Merging (mergeops)

- **Flux merge support** - `/merge/*` endpoints currently reject `model_type: "flux"` (see `api/models/merge.py`'s `field_validator`s) and `metastable.merge` is not wired up for it.

The existing merge pipeline does raw key-by-key tensor merging (`validate_checkpoints()` + `merge_checkpoints()`) assuming two single-file UNet-shaped checkpoints with matching keys/shapes. This doesn't directly apply to Flux:

- Flux checkpoints are transformer-only (`FluxTransformer2DModel`, MMDiT blocks), not a UNet — a different state-dict shape and key-naming convention than `SD15MergePipeline`/`SDXLMergePipeline` expect.
- A "Flux checkpoint" in this backend is just the transformer; the text encoders/VAE/scheduler live in the shared `flux_components_path` directory and aren't part of what gets merged. Any `FluxMergePipeline` would merge only the transformer weights.
- Needs its own `validate_checkpoints()`/`merge_checkpoints()` path (or a generalization of the existing ones) that tolerates the transformer's tensor layout, plus a `FluxMergePipeline` analogous to `SD15MergePipeline`/`SDXLMergePipeline` in `metastable.merge.pipeline` (this work belongs in metastable).
- Once implemented, extend the `model_type` validators in `api/models/merge.py` (`MergeRequest`, `BatchMergeRequest`, `MergeRecipeRequest`) and `api/routers/models.py` to accept `"flux"`.

### Other Flux Follow-ups

- **Quantization** - `bitsandbytes` is already a declared dependency but unused; wiring up NF4 quantization for the transformer and/or T5-XXL encoder would reduce VRAM further beyond `enable_model_cpu_offload()`.
- **FluxFillPipeline** - current Flux inpainting uses `FluxInpaintPipeline` (strength-based re-noising) for consistency with the existing `forward_inpaint()` base-class shape; a dedicated `FluxFillPipeline` may give better inpainting quality and is worth evaluating separately.
- **Shared-components offload hooks** - when two Flux pipelines (e.g. txt2img + inpaint, or two different checkpoints) are simultaneously cache-resident, `enable_model_cpu_offload()` on the second pipeline replaces (rather than stacks) the offload hooks on their shared text encoder/VAE modules, making prefetch ordering suboptimal (not a correctness bug). A future refinement could track offload state on the shared-components dict and avoid redundant `enable_model_cpu_offload()` calls.

### Job Queue & Model Cache Known Issues

- **Flux LoRAs leak through shared text encoders** - all Flux pipelines share one set of text encoders/VAE (metastable's `load_flux_shared_components`). A Flux LoRA that carries text-encoder weights gets fused into those shared modules, so it affects every Flux checkpoint. The model cache's LoRA/VAE state tracking reloads the pipeline, but the reload reuses the same shared modules. Fix belongs in metastable: unfuse/unload LoRAs (`unfuse_lora()` / `unload_lora_weights()`) instead of fusing permanently, or don't share text encoders between pipelines that have LoRAs applied.
- **Model cache key ignores the directory** - `ModelCacheManager._make_cache_key()` uses `Path(model_path).name`, so two checkpoints with the same file name in different subdirectories (`enumerate_models()` is recursive) share a cache entry and lock, and one can be served in place of the other. Fix: key on the resolved full path.
