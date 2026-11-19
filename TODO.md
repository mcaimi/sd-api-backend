## TODO

### Flux / Z-Image Follow-ups

- **Quantization** - `bitsandbytes` is already a declared dependency but unused; wiring up NF4 quantization for the transformer and/or T5-XXL encoder would reduce VRAM further beyond `enable_model_cpu_offload()`.
- **FluxFillPipeline** - current Flux inpainting uses `FluxInpaintPipeline` (strength-based re-noising) for consistency with the existing `forward_inpaint()` base-class shape; a dedicated `FluxFillPipeline` may give better inpainting quality and is worth evaluating separately.
- **Z-Image checkpoint layout** - Flux checkpoints get a `flux_layout` (bundled components) in the models listing, at download time and in the Civitai remote layout check (`api/services/model_layout.py`). Z-Image has no equivalent yet, and loading all-in-one Z-Image files (transformer bundled with Qwen3/VAE) through `from_single_file()` is untested.
- **Z-Image variant hint** - requests must name the variant (`turbo`/`base`); for Civitai downloads the sidecar's `base_model` (`ZImageTurbo`/`ZImageBase`) could pick the default instead.
