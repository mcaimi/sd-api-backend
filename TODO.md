## TODO

Many features still need to be ported over from the [sd-webapp](https://codeberg.org/mcaimi/sd-webapp) project.

The groundwork for everything has been ported over (e.g. simply copied from the old repo :))

This document outlines what it is missing and what will *eventually* come:

### Model Merging Pipeline

- **merge features** - Model merge APIs for `/merge` endpoints (single, batch, and recipe-based) are not yet exposed. 

The merge logic is fully implemented in `libs/stablediffusion/merge.py`, `mergeops.py`, and `funcs.py` (supporting LINEAR, SLERP, ADDITIVE, SUBTRACT methods), and the Pydantic models (`MergeRequest`, `MergeRecipeRequest`, etc.) exist in `api/models/merge.py`, but the API router has not been created to expose these capabilities publicly.

- **Merge Architecture**

This is how I plan to implement it, but this may change over time

### Model Merge Architecture

```
┌──────────────────┐     ┌────────────────────┐
│   MergePipeline   │────>│   MergeConfig      │
│  (ABC)           │     │ ────────────────── │
│ ──────────────── │     │ - method: Enum     │
│ + merge_for_...  │     │ - alpha: float     │
│ + create_recipe()│     │ - device: str      │
│ + exec_recipe()  │     │ - persist_metadata │
└────────┬─────────┘     └────────────────────┘
         │ inherits
   ┌─────┴──────┐
   │            │
┌─────────┐ ┌─────────┐
│ SD15    │ │ SDXL    │
│Merge    │ │ Merge   │
│Pipeline │ │ Pipeline │
└─────────┘ └─────────┘

mergeops (low-level):
  load_checkpoint_dict() → validate_checkpoints() → merge_checkpoints() → save_checkpoint()
  batch_merge_checkpoints() for multi-target
  merge_checkpoints_chunked() for memory efficiency

funcs (tensor ops):
  slerp() | linear() | merge_tensors() | get_random_seed()
```

### Model Merging Call Flow

```
MergePipeline.merge_for_pipeline_generator(base_checkpoint, target_checkpoint, config, output_path)
  → load_checkpoint_dict(base)     → (weights_dict, safetensors_metadata)
  → load_checkpoint_dict(target)   → (weights_dict, safetensors_metadata)
  → validate_checkpoints(base, target)  → assert key match + shape match
  → merge_checkpoints(base, target, config)
     → for each tensor: merge_tensors(tensor_a, tensor_b, method=alpha)
  → save_checkpoint(merged_weights, output_path, metadata)
```

For multi-step recipes:
```
MergePipeline.create_merge_recipe(base_model, [step1, step2, ...])
    → MergeRecipe { base_model, steps: [{target, method, alpha}, ...] }
MergePipeline.execute_merge_recipe(recipe, output_path)
    → for each step: load target → merge current_weights with target → continue
    → save final merged weights
```

### Management UI

I plan to add a web ui:

- monitoring running/canceled/failed/completed jobs
- browse the output directory (look at and download artifacts, both json and png)
- browse the model checkpoints directory (download merged checkpoints)
- TBD