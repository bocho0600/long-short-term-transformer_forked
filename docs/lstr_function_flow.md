# LSTR Code Function Flow

This note explains the code in call-flow order. The project is built around small registries: files import modules for their side effects, classes/functions register themselves, and builder functions later look them up by config names.

## 1. Training Entrypoint

### `main(cfg)` - `tools/train_net.py:16`

Top-level training function. It receives a fully loaded config and performs setup in this order:

1. Calls `setup_environment(cfg)` to choose GPU/CPU and seed randomness.
2. Calls `setup_checkpointer(cfg, phase='train')` to prepare resume/save behavior.
3. Calls `setup_logger(cfg, phase='train')` to log config and epoch results.
4. Builds one `DataLoader` per phase in `cfg.SOLVER.PHASES`, usually `train` and `test`.
5. Calls `build_model(cfg, device)` to construct `LSTRStream`.
6. Calls `build_criterion(cfg, device)` to create losses.
7. Calls `build_optimizer(cfg, model)` to create SGD/Adam/AdamW.
8. Calls `checkpointer.load(model, optimizer)` to resume weights and optimizer state if a checkpoint exists.
9. Calls `build_scheduler(cfg, optimizer, len(data_loaders['train']))`.
10. Calls `do_train(...)`, which dispatches to the LSTR trainer.

The important idea: `main()` does not train directly. It wires together independent components, then hands them to the engine.

### `if __name__ == '__main__'` - `tools/train_net.py:57`

Calls `main(load_cfg())`. This means CLI parsing and YAML config loading happen before the training setup described above.

## 2. Test Entrypoint

### `main(cfg)` - `tools/test_net.py:12`

Top-level test/inference function. It is shorter than training:

1. Calls `setup_environment(cfg)`.
2. Calls `setup_checkpointer(cfg, phase='test')`; in test mode a missing checkpoint raises an error.
3. Calls `setup_logger(cfg, phase='test')`.
4. Builds the model.
5. Loads checkpoint weights with `checkpointer.load(model)`.
6. Calls `do_inference(cfg, model, device, logger)`.

Unlike training, it does not build an optimizer, scheduler, criterion, or train/test data loaders. Inference builds its own dataset later depending on batch or stream mode.

## 3. Config Loading

### `parse_args()` - `utils/parser.py:11`

Defines three CLI inputs:

- `--config_file`: YAML file path.
- `--gpu`: visible GPU ids, stored as a string such as `"0"`.
- `opts`: free-form command-line overrides passed to yacs, for example `SOLVER.BASE_LR 0.0001`.

Returns an argparse namespace.

### `get_cfg()` - `config/defaults.py:134`

Returns a clone of the default yacs config tree. This clone is important because later code mutates the config, for example to fill dataset metadata and computed memory lengths.

### `assert_and_infer_cfg(cfg, args)` - `utils/parser.py:34`

Mutates and validates the config after YAML and CLI overrides have been applied.

Flow:

1. Stores `args.gpu` into `cfg.GPU`.
2. Opens `cfg.DATA.DATA_INFO`, usually `data/data_info.json`.
3. Reads the JSON entry for `cfg.DATA.DATA_NAME`, for example `THUMOS` or `TVSeries`.
4. Fills missing dataset fields: root path, class names, class count, ignore index, metric type, FPS, train sessions, and test sessions.
5. Checks `cfg.INPUT.MODALITY` is one of `visual`, `motion`, or `twostream`.
6. If the model is `LSTR`, converts memory seconds into frame counts:
   - `AGES_MEMORY_LENGTH = AGES_MEMORY_SECONDS * FPS`
   - `LONG_MEMORY_LENGTH = LONG_MEMORY_SECONDS * FPS`
   - `WORK_MEMORY_LENGTH = WORK_MEMORY_SECONDS * FPS`
   - `TOTAL_MEMORY_LENGTH = ages + long + work`
7. Verifies each memory length is divisible by its sample rate.
8. Computes number of sampled frames for each memory region:
   - `LONG_MEMORY_NUM_SAMPLES = LONG_MEMORY_LENGTH // LONG_MEMORY_SAMPLE_RATE`
   - `WORK_MEMORY_NUM_SAMPLES = WORK_MEMORY_LENGTH // WORK_MEMORY_SAMPLE_RATE`
9. Checks LSTR inference mode is `batch` or `stream`.
10. Constructs `cfg.OUTPUT_DIR` from the base output directory plus the config path pieces. If `cfg.SESSION` is set, appends it.

This function is where time-based config becomes actual tensor sizes.

### `load_cfg()` - `utils/parser.py:88`

Config entrypoint. Calls:

1. `parse_args()`
2. `get_cfg()`
3. `cfg.merge_from_file(args.config_file)`
4. `cfg.merge_from_list(args.opts)`
5. `assert_and_infer_cfg(cfg, args)`

Returns the final config object used by training/test.

## 4. Defaults And Data Metadata

### `_C` config tree - `config/defaults.py`

Defines all default settings. Important groups:

- `MODEL`: model name and checkpoint path.
- `MODEL.FEATURE_HEAD`: optional input projection.
- `MODEL.LSTR`: memory lengths, sample rates, transformer heads, feedforward size, dropout, encoder/decoder module definitions, inference mode.
- `MODEL.CRITERIONS`: default loss list, usually `[['MCE', {}]]`.
- `DATA`: dataset metadata, filled from JSON when unset.
- `INPUT`: feature names and target folder.
- `DATA_LOADER`: batch size, workers, pin memory.
- `SOLVER`: epochs, optimizer, LR, scheduler, phases.

### `data/data_info.json`

Maps dataset names to dataset-specific constants. For example:

- `THUMOS`: 22 classes, ignore index 21, metric `AP`, FPS 4.
- `TVSeries`: 31 classes, ignore index -100, metric `cAP`, FPS 4.

The config loader uses this file so individual YAML files do not need to repeat all dataset metadata.

## 5. Environment, Logging, Checkpoints

### `setup_random_seed(seed)` - `utils/env.py:16`

Seeds Python `random`, NumPy, and PyTorch. If CUDA is available, also seeds CUDA. Sets:

- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`

This favors reproducibility over maximum cuDNN autotuning speed.

### `setup_environment(cfg)` - `utils/env.py:27`

Sets `CUDA_VISIBLE_DEVICES` from `cfg.GPU`. Creates `torch.device('cuda')` if CUDA is available, otherwise CPU. Calls `setup_random_seed(cfg.SEED)` when a seed is configured. Returns the device.

### `setup_checkpointer(cfg, phase)` - `utils/checkpointer.py:43`

Constructs `Checkpointer(cfg, phase)`.

### `Checkpointer.__init__(cfg, phase)` - `utils/checkpointer.py:13`

Loads `cfg.MODEL.CHECKPOINT` using `_load_checkpoint()`.

Behavior:

- If checkpoint exists and phase is `train`, adds the stored checkpoint epoch to `cfg.SOLVER.START_EPOCH`.
- If checkpoint does not exist and phase is not `train`, raises `RuntimeError`.
- Stores `cfg.OUTPUT_DIR` for later saving.

### `Checkpointer._load_checkpoint(checkpoint)` - `utils/checkpointer.py:37`

If `checkpoint` is a real file, returns `torch.load(checkpoint, map_location='cpu')`. Otherwise returns `None`.

Expected checkpoint dict:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`

### `Checkpointer.load(model, optimizer=None)` - `utils/checkpointer.py:24`

If a checkpoint was found:

1. Loads model weights using `model.load_state_dict(...)`.
2. Loads optimizer state if an optimizer was supplied.

In test mode, only model weights are loaded.

### `Checkpointer.save(epoch, model, optimizer)` - `utils/checkpointer.py:30`

Saves a checkpoint to `OUTPUT_DIR/epoch-{epoch}.pth`. If multiple CUDA devices are visible, saves `model.module.state_dict()` because the model may be wrapped in `nn.DataParallel`; otherwise saves `model.state_dict()`.

### `setup_logger(cfg, phase, quiet=False)` - `utils/logger.py:12`

Creates/reuses logger named `rekognition`. Sets log level from `cfg.VERBOSE`. Adds:

- stdout stream handler.
- file handler.

For training, log path is `cfg.OUTPUT_DIR/log.txt`. For test, log path is checkpoint path with `.txt` extension. Unless `quiet=True`, logs the full config.

## 6. Registry System

### `_register_generic(module_dict, module_name, module)` - `utils/registry.py:4`

Asserts the name is unused, then stores `module_dict[module_name] = module`.

### `Registry.register(module_name, module=None)` - `utils/registry.py:14`

Supports two registration styles:

- Direct call: `registry.register('Name', SomeClass)`
- Decorator: `@registry.register('Name')`

Most of this repo uses decorators. Example: `@registry.register('LSTR')` registers `LSTRStream` as the model class.

## 7. Import Side Effects

### `datasets/__init__.py`

Exports `build_dataset` and `build_data_loader`, then imports `perframe_data_layers`. That import is necessary because dataset classes register themselves when the module is imported.

### `models/__init__.py`

Exports `build_model`, then imports `lstr`. That import registers `LSTRStream`.

### `engines/__init__.py`

Exports `do_train` and `do_inference`, then imports LSTR trainer and inference modules. Those imports register the LSTR train/inference functions.

## 8. Data Loading

### `build_dataset(cfg, phase, tag='')` - `datasets/datasets.py:16`

Builds a registry key:

```text
cfg.MODEL.MODEL_NAME + tag + cfg.DATA.DATA_NAME
```

Examples:

- Training THUMOS: `LSTRTHUMOS`
- Batch inference THUMOS: `LSTRBatchInferenceTHUMOS`

Looks up the dataset class in `DATA_LAYERS` and instantiates it.

### `build_data_loader(cfg, phase)` - `datasets/datasets.py:21`

Wraps the dataset in a PyTorch `DataLoader`.

Uses:

- `batch_size = cfg.DATA_LOADER.BATCH_SIZE`
- `shuffle = True` only for training
- `num_workers`
- `pin_memory`

## 9. Training Dataset

### `LSTRDataLayer.__init__(cfg, phase='train')` - `datasets/perframe_data_layers.py:18`

Reads dataset and memory config into instance variables:

- data folders for visual, motion, and target arrays
- session list for train or test
- long/work memory lengths, sample rates, and sample counts
- `self.training`

Then calls `_init_dataset()` to build the sample index list.

### `LSTRDataLayer._init_dataset()` - `datasets/perframe_data_layers.py:37`

Builds `self.inputs`, a list of clips. For each session:

1. Loads per-frame targets from `target_perframe/{session}.npy`.
2. Chooses a random starting offset during training, or 0 during test.
3. Slides non-overlapping work windows of length `work_memory_length`.
4. Stores `[session, work_start, work_end, target_slice]`.

This function does not load visual/motion features. It only indexes clip windows and target slices.

### `LSTRDataLayer.shuffle()` - `datasets/perframe_data_layers.py:34`

Calls `_init_dataset()` again. During training this picks new random work-window offsets for the next epoch.

### `LSTRDataLayer.segment_sampler(start, end, num_samples)` - `datasets/perframe_data_layers.py:49`

Uses `np.linspace(start, end, num_samples)` to choose evenly spread long-memory frame indices. It is used for training long memory.

Because `linspace` includes both ends and can produce fractional values before integer conversion, this gives broad coverage of the long-memory interval.

### `LSTRDataLayer.uniform_sampler(start, end, num_samples, sample_rate)` - `datasets/perframe_data_layers.py:53`

Creates `np.arange(start, end + 1)[::sample_rate]`. If not enough indices exist, prepends zeros as padding. Used in test for deterministic long-memory sampling.

### `LSTRDataLayer.__getitem__(index)` - `datasets/perframe_data_layers.py:60`

Called by the `DataLoader` for each sample.

Flow:

1. Reads `[session, work_start, work_end, target]` from `self.inputs[index]`.
2. Memory-maps visual and motion `.npy` feature arrays. This avoids loading entire videos into RAM.
3. Subsamples target labels by `work_memory_sample_rate`.
4. Builds work-memory indices from `work_start` to `work_end - 1`, clips negative values to 0, then subsamples.
5. Reads work visual/motion features.
6. If long memory is enabled:
   - Computes long window `[work_start - long_memory_length, work_start - 1]`, clipped at 0.
   - Uses `segment_sampler()` during training.
   - Uses `uniform_sampler()` during test.
   - Reads long visual/motion features.
   - Builds `memory_key_padding_mask`, where padded positions get `-inf` so attention ignores them.
7. Concatenates long memory before work memory.
8. Converts visual, motion, target, and optional mask to `torch.float32`.
9. Returns either:
   - `(fusion_visual, fusion_motion, memory_key_padding_mask, target)`
   - or `(fusion_visual, fusion_motion, target)` when long memory is disabled.

Shape intuition:

- `fusion_visual`: `[long_samples + work_samples, visual_feature_dim]` per sample before batching.
- DataLoader batches it to `[B, total_samples, visual_feature_dim]`.
- `target`: `[work_samples, num_classes]` per sample, batched to `[B, work_samples, num_classes]`.

### `LSTRDataLayer.__len__()` - `datasets/perframe_data_layers.py:123`

Returns number of indexed work windows in `self.inputs`.

## 10. Batch Inference Dataset

### `LSTRBatchInferenceDataLayer.__init__(cfg, phase='test')` - `datasets/perframe_data_layers.py:131`

Similar to `LSTRDataLayer`, but only supports `phase='test'`. It creates one sample per sliding frame-level query:

- `work_start` ranges from 0 to `num_frames`.
- `work_end` ranges from `work_memory_length` to `num_frames`.

Stores `[session, work_start, work_end, target_slice, num_frames]`.

This lets batch inference produce a prediction for every frame by repeatedly moving the work window by one frame.

### `LSTRBatchInferenceDataLayer.uniform_sampler(...)` - `datasets/perframe_data_layers.py:156`

Same logic as the training dataset's deterministic sampler: stride through the long window and left-pad with zeros if needed.

### `LSTRBatchInferenceDataLayer.__getitem__(index)` - `datasets/perframe_data_layers.py:163`

Builds one inference sample. It follows the same long/work memory construction as test mode in `LSTRDataLayer`, but returns extra metadata:

- `session`
- `work_indices`
- `num_frames`

These are needed so inference can place predictions back into full-video arrays.

Return forms:

- `(fusion_visual, fusion_motion, mask, target, session, work_indices, num_frames)`
- or `(fusion_visual, fusion_motion, target, session, work_indices, num_frames)`.

### `LSTRBatchInferenceDataLayer.__len__()` - `datasets/perframe_data_layers.py:224`

Returns number of frame-query windows.

## 11. Model Construction

### `build_model(cfg, device=None)` - `models/models.py:11`

Looks up model class from `META_ARCHITECTURES[cfg.MODEL.MODEL_NAME]`. For LSTR this is `LSTRStream`. Then:

1. Instantiates the model.
2. Imports `weights_init`.
3. Applies `weights_init` recursively to every submodule.
4. Moves model to `device`.

### `weights_init(m)` - `models/weights_init.py:9`

Initialization function passed to `model.apply()`.

Layer behavior:

- `nn.Linear`: Kaiming uniform.
- `nn.Conv1d` and `nn.ConvTranspose1d`: normal weights and normal bias.
- `nn.Conv2d` and `nn.ConvTranspose2d`: Xavier normal weights and normal bias.
- `nn.BatchNorm1d` and `nn.BatchNorm2d`: weights near 1, bias 0.

## 12. Feature Head

### `BaseFeatureHead.__init__(cfg)` - `models/feature_head.py:26`

Builds the feature fusion/projection block for THUMOS or TVSeries.

Flow:

1. Reads modality:
   - `visual`: use visual only.
   - `motion`: use motion only.
   - `twostream`: concatenate visual and motion.
2. Looks up feature dimensions from `FEATURE_SIZES`.
3. Sets `self.d_model` to the fusion feature size.
4. If `MODEL.FEATURE_HEAD.LINEAR_ENABLED` is true, builds:

```text
Linear(fusion_size -> LINEAR_OUT_FEATURES)
ReLU
```

If linear is disabled, uses `nn.Identity()`.

### `BaseFeatureHead.forward(visual_input, motion_input)` - `models/feature_head.py:56`

Combines inputs according to modality:

- concatenates visual and motion for twostream
- passes through only one stream for single-stream modes
- applies `self.input_linear`

Returns features shaped like `[B, T, d_model]`.

### `build_feature_head(cfg)` - `models/feature_head.py:67`

Looks up `FEATURE_HEADS[cfg.DATA.DATA_NAME]` and returns the dataset-specific feature head class. In this repo THUMOS and TVSeries both use `BaseFeatureHead`.

## 13. LSTR Model

### `LSTR.__init__(cfg)` - `models/lstr.py:14`

Builds the LSTR architecture.

Flow:

1. Reads long memory sample count.
2. If long memory exists, builds `feature_head_long`.
3. Reads work memory sample count.
4. If work memory exists, builds `feature_head_work`.
5. Sets transformer dimensions: `d_model`, heads, feedforward size, dropout, activation, number of classes.
6. Builds sinusoidal `PositionalEncoding`.
7. Builds long-memory encoder modules if long memory is enabled.
8. Builds work-memory decoder modules.
9. Builds classifier `Linear(d_model -> num_classes)`.

Encoder module config format:

```text
[num_queries, num_layers, use_layer_norm]
```

If `num_queries != -1`, the code builds a `TransformerDecoder`: learned query tokens attend to long memory and compress it. If `num_queries == -1`, it builds a `TransformerEncoder`: long memory attends to itself.

Decoder module config format:

```text
[ignored_or_minus_one, num_layers, use_layer_norm]
```

If long memory exists, the decoder is a `TransformerDecoder`: work memory attends to encoded long memory. If long memory does not exist, it is a `TransformerEncoder`: work memory attends only to itself with a causal mask.

### `LSTR.forward(visual_inputs, motion_inputs, memory_key_padding_mask=None)` - `models/lstr.py:81`

Normal forward pass used by training and batch inference.

Input shape after batching:

- `visual_inputs`: `[B, long_samples + work_samples, visual_dim]`
- `motion_inputs`: `[B, long_samples + work_samples, motion_dim]`
- `memory_key_padding_mask`: optional `[B, long_samples]`

Flow:

1. If long memory is enabled:
   - Slices long visual/motion inputs.
   - Fuses/projects them through `feature_head_long`.
   - Transposes from `[B, T, D]` to transformer format `[T, B, D]`.
   - Adds positional encoding.
   - Builds learned encoder queries when configured.
   - Sends long memory through each encoder module.
   - The first encoder module receives `memory_key_padding_mask`, so padded long-memory positions are ignored.
2. If work memory is enabled:
   - Slices work visual/motion inputs.
   - Fuses/projects them through `feature_head_work`.
   - Transposes to `[T, B, D]`.
   - Adds positional encoding with padding offset equal to `long_memory_num_samples`, so work positions come after long-memory positions.
   - Builds a causal square mask with `generate_square_subsequent_mask()`.
   - If long memory exists, calls decoder with `work_memories` as target and encoded long memory as `memory`.
   - If long memory does not exist, calls encoder over work memory with causal mask.
3. Classifies each work-memory output vector with `self.classifier`.
4. Transposes scores back to `[B, work_samples, num_classes]`.

This model predicts one class distribution per work-memory sampled frame.

### `LSTRStream.__init__(cfg)` - `models/lstr.py:146`

Calls `LSTR.__init__()`, then creates stream-inference caches:

- `long_memories_cache`
- `compressed_long_memories_cache`

### `LSTRStream.stream_inference(...)` - `models/lstr.py:155`

Online inference path. It avoids recomputing all long-memory attention from scratch every frame.

Inputs are already separated:

- `long_visual_inputs`, `long_motion_inputs`: either a new long-memory frame or `None`.
- `work_visual_inputs`, `work_motion_inputs`: current work window.
- `memory_key_padding_mask`: padding mask for long memory.

Flow:

1. Requires long memory and at least one encoder module.
2. If new long-memory inputs are provided:
   - Projects them with `feature_head_long`.
   - Appends them to `long_memories_cache`, dropping the oldest cached item.
   - Uses fixed positional encodings for the whole long-memory cache.
   - Builds encoder queries.
   - Runs the first encoder module through its stream path.
   - Stores compressed result in `compressed_long_memories_cache`.
   - Runs remaining encoder modules normally.
3. If no new long-memory input is provided:
   - Reuses `compressed_long_memories_cache`.
   - Runs only later encoder modules if present.
4. Projects current work memory.
5. Builds causal work mask.
6. Runs decoder against long memory.
7. Classifies and returns `[B, work_samples, num_classes]`.

## 14. Positional Encoding And Transformer Utilities

### `PositionalEncoding.__init__(d_model, dropout=0.1, max_len=5000)` - `transformer/position_encoding.py:12`

Creates sinusoidal positional encodings:

- sine on even dimensions
- cosine on odd dimensions

Stores them as a non-trainable buffer named `pe`.

### `PositionalEncoding.forward(x, padding=0)` - `transformer/position_encoding.py:25`

Adds position encodings to `x`. The `padding` offset lets work memory use later positions after long memory. Applies dropout and returns the result.

### `layer_norm(d_model, condition=True)` - `transformer/utils.py:9`

Returns `nn.LayerNorm(d_model)` if `condition` is true, otherwise `None`.

### `generate_square_subsequent_mask(sz)` - `transformer/utils.py:13`

Creates a causal attention mask of shape `[sz, sz]`.

- Allowed positions get `0`.
- Future positions get `-inf`.

This prevents a work-memory frame from attending to future work-memory frames.

## 15. Transformer Stack

### `Transformer.__init__(...)` - `transformer/transformer.py:15`

Generic encoder-decoder wrapper, similar to PyTorch's transformer. It can use custom encoder/decoder or build default `TransformerEncoder` and `TransformerDecoder`.

LSTR does not use this top-level class directly in its main architecture, but the components below are used.

### `Transformer.forward(src, tgt, ...)` - `transformer/transformer.py:37`

Checks batch size and feature dimension compatibility. Runs:

1. `self.encoder(src, ...)`
2. `self.decoder(tgt, memory, ...)`

Returns decoder output.

### `TransformerEncoder.__init__(encoder_layer, num_layers, norm=None)` - `transformer/transformer.py:55`

Clones one encoder layer `num_layers` times using `_get_clones()`. Stores optional final norm.

### `TransformerEncoder.forward(src, src_mask=None, src_key_padding_mask=None)` - `transformer/transformer.py:62`

Passes source through each encoder layer. Applies final norm if present.

### `TransformerDecoder.__init__(decoder_layer, num_layers, norm=None)` - `transformer/transformer.py:77`

Clones one decoder layer `num_layers` times. Stores optional final norm.

### `TransformerDecoder.forward(tgt, memory, ...)` - `transformer/transformer.py:103`

Passes target through each decoder layer. Each layer performs:

1. target self-attention
2. cross-attention to memory
3. feed-forward network

Applies final norm if present.

### `TransformerDecoder.stream_inference(...)` - `transformer/transformer.py:84`

Stream-only decoder path. Requires exactly one layer. Calls `stream_inference()` on that single decoder layer, then applies final norm.

### `TransformerEncoderLayer.__init__(...)` - `transformer/transformer.py:122`

Builds one transformer encoder layer:

- self-attention
- `Linear(d_model -> dim_feedforward)`
- activation
- dropout
- `Linear(dim_feedforward -> d_model)`
- residual connections
- two layer norms

### `TransformerEncoderLayer.forward(src, src_mask=None, src_key_padding_mask=None)` - `transformer/transformer.py:144`

Runs:

1. Self-attention over `src`.
2. Residual add and norm.
3. Feed-forward MLP.
4. Residual add and norm.

Returns encoded source.

### `TransformerDecoderLayer.__init__(...)` - `transformer/transformer.py:157`

Builds one decoder layer:

- target self-attention
- cross-attention from target to memory
- feed-forward MLP
- three residual/norm blocks
- `tgt_cache` for stream inference

### `TransformerDecoderLayer.forward(tgt, memory, ...)` - `transformer/transformer.py:206`

Runs:

1. Self-attention over `tgt` with optional causal mask.
2. Residual add and norm.
3. Cross-attention where query is target and key/value are memory.
4. Residual add and norm.
5. Feed-forward MLP.
6. Residual add and norm.

### `TransformerDecoderLayer.stream_inference(...)` - `transformer/transformer.py:187`

Stream path:

1. Computes and caches target self-attention output the first time.
2. Reuses cached target state later.
3. Calls stream cross-attention against long memory.
4. Runs feed-forward and norms.

### `_get_clones(module, N)` - `transformer/transformer.py:222`

Returns an `nn.ModuleList` containing `N` deep copies of the given module.

### `_get_activation_fn(activation)` - `transformer/transformer.py:226`

Maps string names to activation functions:

- `relu` -> `F.relu`
- `gelu` -> `F.gelu`

Raises an error for unknown activations.

## 16. Attention

### `DotProductAttention.__init__(dropout=0.0)` - `transformer/multihead_attention.py:11`

Stores dropout probability.

### `DotProductAttention.forward(q, k, v, attn_mask=None)` - `transformer/multihead_attention.py:16`

Computes scaled-dot-product attention after Q/K/V have already been projected and split into heads:

1. Computes `q @ k^T`.
2. Adds attention mask if present.
3. Applies softmax over key positions.
4. Applies dropout.
5. Multiplies by `v`.

Returns attention output.

### `DotProductAttentionStream.__init__(dropout=0.0)` - `transformer/multihead_attention.py:32`

Extends dot-product attention with caches:

- `k_weights_cache`
- `k_pos_weights_cache`

### `DotProductAttentionStream.stream_inference(q, k, v, k_pos, v_pos, attn_mask=None)` - `transformer/multihead_attention.py:41`

Stream attention over a sliding long-memory cache.

If cached weights exist, it computes only the new last key contribution and shifts the old weights left. It reuses positional key weights. Then it softmaxes, drops out, and multiplies by `v + v_pos`.

### `MultiheadAttention.__init__(embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None)` - `transformer/multihead_attention.py:67`

Custom multi-head attention module.

Builds:

- combined Q/K/V projection matrix of shape `[3 * embed_dim, embed_dim]`
- combined Q/K/V bias
- output projection `Linear(embed_dim, embed_dim)`
- `DotProductAttention`

Only supports Q, K, and V with the same embedding dimension.

### `MultiheadAttention.forward(q, k, v, attn_mask=None, key_padding_mask=None)` - `transformer/multihead_attention.py:97`

Flow:

1. Reads sequence length, batch size, embedding dimension.
2. Checks `embed_dim` is divisible by `num_heads`.
3. Projects q, k, v using slices of `in_proj_weight`.
4. Scales q by `head_dim ** -0.5`.
5. Reshapes q/k/v from `[T, B, D]` to `[B * heads, T, head_dim]`.
6. Expands `attn_mask` and `key_padding_mask` to all heads.
7. Combines masks if both exist.
8. Calls `DotProductAttention.forward()`.
9. Reassembles heads back to `[T, B, D]`.
10. Applies output projection.

Returns `(attn_output, None)`. The second value mirrors PyTorch's attention API but attention weights are not returned.

### `MultiheadAttentionStream.__init__(...)` - `transformer/multihead_attention.py:162`

Extends `MultiheadAttention` and replaces dot-product attention with `DotProductAttentionStream`. Adds caches for q/k/v and positional projections.

### `MultiheadAttentionStream.stream_inference(q, k, v, pos, attn_mask=None, key_padding_mask=None)` - `transformer/multihead_attention.py:176`

Stream version of multi-head attention.

Flow:

1. Reuses cached q projection if available.
2. For k and v:
   - If cache exists, project only the newest final item and shift the cache.
   - If no cache exists, project the full sequence and positional encodings.
3. Scales q.
4. Splits q/k/v and positional k/v into heads.
5. Expands masks.
6. Calls `DotProductAttentionStream.stream_inference()`.
7. Reassembles heads and applies output projection.

This is the lower-level mechanism that makes stream inference faster than recomputing all long-memory attention at every frame.

## 17. Losses

### `BinaryCrossEntropyLoss.__init__(reduction='mean', ignore_index=-100)` - `criterions/criterions.py:18`

Wraps `nn.BCEWithLogitsLoss`. The `ignore_index` argument is accepted for API consistency but not used.

### `BinaryCrossEntropyLoss.forward(input, target)` - `criterions/criterions.py:23`

Returns BCE-with-logits loss.

### `SingleCrossEntropyLoss.__init__(reduction='mean', ignore_index=-100)` - `criterions/criterions.py:30`

Wraps `nn.CrossEntropyLoss`, passing through reduction and ignore index.

### `SingleCrossEntropyLoss.forward(input, target)` - `criterions/criterions.py:36`

Returns single-label cross entropy.

### `MultipCrossEntropyLoss.__init__(reduction='mean', ignore_index=-100)` - `criterions/criterions.py:43`

Stores reduction mode and ignore index. This is the default LSTR loss under registry name `MCE`.

### `MultipCrossEntropyLoss.forward(input, target)` - `criterions/criterions.py:49`

Implements soft/multi-label cross entropy:

1. Creates `LogSoftmax(dim=1)`.
2. If `ignore_index >= 0`:
   - Excludes the ignore class column from the class loss.
   - Computes `sum(-target * log_probs)` over remaining classes.
   - Filters out samples where the ignore class target is 1.
3. If no ignore index:
   - Computes the same loss over all classes.
4. Reduces by mean, sum, or returns unreduced values.

### `build_criterion(cfg, device=None)` - `criterions/criterions.py:73`

Iterates `cfg.MODEL.CRITERIONS`. For each `[name, params]`:

1. Looks up loss class in `CRITERIONS`.
2. Adds `cfg.DATA.IGNORE_INDEX` to params if missing.
3. Instantiates and moves loss to device.
4. Stores in a dictionary by name.

Returns for default config: `{'MCE': MultipCrossEntropyLoss(...)}`.

## 18. Optimizer And Scheduler

### `build_optimizer(cfg, model)` - `optimizers/optimizers.py:9`

Creates one optimizer over `model.parameters()`.

Supported names:

- `sgd`
- `adam`
- `adamw`

All include `initial_lr` in the param group so the custom scheduler can resume correctly.

### `_get_warmup_factor_at_iter(warmup_method, this_iter, warmup_iters, warmup_factor)` - `optimizers/lr_scheduler.py:12`

Returns LR multiplier during warmup:

- after warmup: `1.0`
- constant warmup: `warmup_factor`
- linear warmup: linearly increases from `warmup_factor` to `1.0`

### `MultiStepLR.__init__(optimizer, milestones, gamma=0.1, last_epoch=-1)` - `optimizers/lr_scheduler.py:30`

Stores sorted iteration milestones and decay factor.

### `MultiStepLR.get_lr()` - `optimizers/lr_scheduler.py:43`

Computes:

```text
base_lr * gamma ** number_of_passed_milestones
```

The scheduler uses iterations, not epochs.

### `MultiStepLR._compute_values()` - `optimizers/lr_scheduler.py:50`

Compatibility method for newer PyTorch scheduler internals. Returns `get_lr()`.

### `WarmupMultiStepLR.__init__(...)` - `optimizers/lr_scheduler.py:57`

Stores milestones, gamma, warmup factor, warmup iterations, and warmup method.

### `WarmupMultiStepLR.get_lr()` - `optimizers/lr_scheduler.py:76`

Computes multi-step LR multiplied by the current warmup factor.

### `WarmupMultiStepLR._compute_values()` - `optimizers/lr_scheduler.py:90`

Returns `get_lr()`.

### `CosineLR.__init__(optimizer, max_iters, last_epoch=-1)` - `optimizers/lr_scheduler.py:97`

Stores total number of training iterations.

### `CosineLR.get_lr()` - `optimizers/lr_scheduler.py:104`

Computes cosine annealing:

```text
base_lr * 0.5 * (1 + cos(pi * iter / max_iters))
```

### `CosineLR._compute_values()` - `optimizers/lr_scheduler.py:112`

Returns `get_lr()`.

### `WarmupCosineLR.__init__(...)` - `optimizers/lr_scheduler.py:119`

Stores max iterations and warmup settings.

### `WarmupCosineLR.get_lr()` - `optimizers/lr_scheduler.py:133`

Computes cosine LR multiplied by warmup factor.

### `WarmupCosineLR._compute_values()` - `optimizers/lr_scheduler.py:148`

Returns `get_lr()`.

### `build_scheduler(cfg, optimizer, num_iters_per_epoch)` - `optimizers/lr_scheduler.py:153`

Builds an iteration-based scheduler.

Flow:

1. Computes `last_epoch = (START_EPOCH - 1) * num_iters_per_epoch`; here PyTorch's `last_epoch` variable is actually last iteration.
2. For multistep variants, converts milestone epochs into iteration milestones.
3. For warmup variants, converts warmup epochs into warmup iterations.
4. Instantiates one of:
   - `MultiStepLR`
   - `WarmupMultiStepLR`
   - `CosineLR`
   - `WarmupCosineLR`

## 19. Training Engine

### `do_train(cfg, data_loaders, model, criterion, optimizer, scheduler, device, checkpointer, logger)` - `engines/engines.py:10`

Dispatches training by model name:

```text
TRAINERS[cfg.MODEL.MODEL_NAME](...)
```

For LSTR this calls `do_lstr_train()`.

### `do_lstr_train(...)` - `engines/lstr/lstr_trainer.py:9`

Thin wrapper registered under `LSTR`. Calls `do_perframe_det_train(...)`.

### `do_perframe_det_train(...)` - `engines/base_trainers/perframe_det_trainer.py:13`

Main training loop.

Setup:

1. If more than one CUDA device is visible, wraps model with `nn.DataParallel`.

Epoch flow:

1. Loops from `START_EPOCH` to `START_EPOCH + NUM_EPOCHS - 1`.
2. Resets train/test loss accumulators and evaluation lists.
3. For each phase in `cfg.SOLVER.PHASES`:
   - Sets `training = phase == 'train'`.
   - Calls `model.train(training)`.
   - Uses `torch.set_grad_enabled(training)`.
4. For each batch:
   - Reads batch size from `data[0]`.
   - Treats last item as target.
   - Moves target to device.
   - Moves all input tensors to device and calls `model(*inputs)`.
   - Reshapes scores from `[B, W, C]` to `[B * W, C]`.
   - Reshapes target the same way.
   - Computes `criterion['MCE'](det_score, det_target)`.
   - Accumulates loss multiplied by batch size.
   - Updates tqdm display with LR and loss.
5. If training:
   - `optimizer.zero_grad()`
   - `det_loss.backward()`
   - `optimizer.step()`
   - `scheduler.step()`
6. If testing:
   - Applies softmax over classes.
   - Adds predictions and targets to evaluation lists.
7. After all phases:
   - Logs train loss.
   - If test phase exists, calls `compute_result['perframe'](...)`.
   - Logs test loss and mAP.
   - Saves checkpoint.
   - Calls `data_loaders['train'].dataset.shuffle()` for new random offsets next epoch.

## 20. Inference Engine

### `do_inference(cfg, model, device, logger)` - `engines/engines.py:31`

Dispatches inference by model name:

```text
INFERENCES[cfg.MODEL.MODEL_NAME](...)
```

For LSTR this calls `do_lstr_batch_inference()`.

### `do_lstr_batch_inference(cfg, model, device, logger)` - `engines/lstr/lstr_inference.py:18`

Despite its name, this function chooses between two inference modes:

- if `cfg.MODEL.LSTR.INFERENCE_MODE == 'stream'`, calls `do_lstr_stream_inference()`
- otherwise calls `do_perframe_det_batch_inference()`

### `do_perframe_det_batch_inference(cfg, model, device, logger)` - `engines/base_inferences/perframe_det_batch_inference.py:16`

Batch inference over sliding windows.

Flow:

1. Sets model to eval mode.
2. Builds `LSTRBatchInferenceDataLayer` through `build_dataset(..., tag='BatchInference')`.
3. Uses batch size `cfg.DATA_LOADER.BATCH_SIZE * 16`.
4. Creates dictionaries for predictions and targets keyed by session.
5. With `torch.no_grad()`:
   - Calls model on tensor inputs.
   - Applies softmax.
   - Uses session and work indices to place predictions into full-video arrays.
   - For the first window, writes all work-window predictions.
   - For later windows, writes only the newest final-frame prediction.
6. Saves a pickle next to checkpoint containing config, per-frame scores, and targets.
7. Calls `compute_result['perframe'](...)`.
8. Logs mean AP/cAP.

### `do_lstr_stream_inference(cfg, model, device, logger)` - `engines/lstr/lstr_inference.py:34`

Online-style inference for a single test video.

Flow:

1. Sets model to eval mode.
2. Defines helper `to_device(x, dtype=np.float32)` to convert NumPy arrays to batched device tensors.
3. Reads memory config values.
4. Requires exactly one test session.
5. Loads visual features, motion features, and targets.
6. Slides the work window one frame at a time.
7. For each step:
   - Builds work-memory inputs.
   - Updates long-memory inputs only when a new frame aligns with `long_memory_sample_rate`.
   - Otherwise passes `None` for long inputs, causing cached compressed memory to be reused.
   - Builds long-memory padding mask.
   - Calls `model.stream_inference(...)`.
   - Applies softmax.
   - Stores all scores for the first window and only the newest score for later windows.
8. Computes per-frame result for the video.
9. Logs mAP and runtime.

Note: the code mutates `target = target[::work_memory_sample_rate]` inside the loop, which is unusual because it repeatedly subsamples the same variable. With the default work sample rate of 1 this has no effect.

## 21. Evaluation

### `eval_perframe(cfg, ground_truth, prediction, **kwargs)` - `evaluation/evalution.py:14`

Registered as `compute_result['perframe']`.

Reads class names, ignore index, metric type, and postprocessing function from config unless overridden. Calls `perframe_average_precision(...)`.

### `eval_perstage(cfg, ground_truth, prediction, **kwargs)` - `evaluation/evalution.py:30`

Registered as `compute_result['perstage']`. Same setup as per-frame evaluation, but calls `perstage_average_precision(...)`.

### `postprocessing(data_name)` - `evaluation/postprocessing.py:7`

Returns dataset-specific postprocessing. Only THUMOS has one in this repo.

### `thumos_postprocessing(ground_truth, prediction, smooth=False, switch=False)` - `evaluation/postprocessing.py:9`

Optional THUMOS logic:

- If `smooth=True`, applies a 5-frame local max smoothing over predictions.
- If `switch=True`, maps strong CliffDiving scores into Diving scores.
- Always removes frames where ambiguous class index 21 is active.

Returns filtered ground truth and predictions.

### `calibrated_average_precision_score(y_true, y_score)` - `evaluation/metrics.py:11`

Computes cAP, mainly for TVSeries. It sorts frames by predicted score, accumulates true positives and false positives, rescales false positives by the background/action ratio, and averages calibrated precision at positive frames.

### `perframe_average_precision(...)` - `evaluation/metrics.py:26`

Computes frame-level AP/cAP.

Flow:

1. Converts inputs to NumPy arrays.
2. Applies postprocessing if provided.
3. Selects metric function: sklearn AP or local cAP.
4. Ignores background class 0 and `ignore_index`.
5. For each remaining class with at least one positive ground-truth frame, computes AP/cAP.
6. Returns ordered dict with per-class AP and mean AP.

### `get_stage_pred_scores(gt_targets, pred_scores, perc_s, perc_e)` - `evaluation/metrics.py:66`

Helper for stage-wise AP. It extracts action segments from a binary class target sequence, then keeps only a percentage slice of each action instance, such as 0-10 percent or 40-50 percent.

Returns stage-specific ground truth and prediction arrays.

### `perstage_average_precision(...)` - `evaluation/metrics.py:92`

Computes AP/cAP separately for ten temporal stages of action instances:

- 0-10 percent
- 10-20 percent
- ...
- 90-100 percent

For each stage and class, it calls `get_stage_pred_scores()` and computes AP/cAP. Returns mean AP per stage.

## 22. Standalone Eval Tools

### `eval_perframe(pred_scores_file)` - `tools/eval/eval_perframe.py:14`

Loads the pickle saved by batch inference, extracts config/targets/predictions, concatenates sessions, calls `compute_result['perframe'](...)`, and logs mAP/cAP.

### `eval_perstage(pred_scores_file)` - `tools/eval/eval_perstage.py:14`

Loads the same pickle, calls `compute_result['perstage'](...)`, and logs each stage's mean AP/cAP.

## 23. Full Training Call Tree

```text
tools/train_net.py
└── main(load_cfg())
    ├── load_cfg()
    │   ├── parse_args()
    │   ├── get_cfg()
    │   ├── cfg.merge_from_file()
    │   ├── cfg.merge_from_list()
    │   └── assert_and_infer_cfg()
    ├── setup_environment()
    │   └── setup_random_seed()
    ├── setup_checkpointer()
    │   └── Checkpointer.__init__()
    │       └── Checkpointer._load_checkpoint()
    ├── setup_logger()
    ├── build_data_loader() for each phase
    │   └── build_dataset()
    │       └── LSTRDataLayer.__init__()
    │           └── LSTRDataLayer._init_dataset()
    ├── build_model()
    │   ├── LSTRStream.__init__()
    │   │   └── LSTR.__init__()
    │   │       ├── build_feature_head()
    │   │       │   └── BaseFeatureHead.__init__()
    │   │       ├── PositionalEncoding.__init__()
    │   │       ├── TransformerDecoder/Encoder modules
    │   │       │   ├── TransformerDecoderLayer.__init__()
    │   │       │   ├── TransformerEncoderLayer.__init__()
    │   │       │   └── MultiheadAttentionStream.__init__()
    │   │       └── nn.Linear classifier
    │   └── model.apply(weights_init)
    ├── build_criterion()
    │   └── MultipCrossEntropyLoss.__init__()
    ├── build_optimizer()
    ├── Checkpointer.load()
    ├── build_scheduler()
    │   └── WarmupCosineLR/MultiStepLR/etc.
    └── do_train()
        └── do_lstr_train()
            └── do_perframe_det_train()
                ├── per batch: LSTR.forward()
                │   ├── BaseFeatureHead.forward()
                │   ├── PositionalEncoding.forward()
                │   ├── TransformerDecoder/Encoder.forward()
                │   │   ├── TransformerDecoderLayer.forward()
                │   │   ├── TransformerEncoderLayer.forward()
                │   │   ├── MultiheadAttention.forward()
                │   │   └── DotProductAttention.forward()
                │   └── classifier
                ├── MultipCrossEntropyLoss.forward()
                ├── optimizer.step()
                ├── scheduler.step()
                │   └── get_lr()
                ├── compute_result['perframe']()
                │   └── perframe_average_precision()
                │       └── calibrated_average_precision_score() if cAP
                ├── Checkpointer.save()
                └── LSTRDataLayer.shuffle()
                    └── LSTRDataLayer._init_dataset()
```

## 24. Full Batch Inference Call Tree

```text
tools/test_net.py
└── main(load_cfg())
    ├── setup_environment()
    ├── setup_checkpointer(phase='test')
    ├── setup_logger(phase='test')
    ├── build_model()
    ├── Checkpointer.load(model)
    └── do_inference()
        └── do_lstr_batch_inference()
            └── do_perframe_det_batch_inference()
                ├── build_dataset(tag='BatchInference')
                │   └── LSTRBatchInferenceDataLayer.__init__()
                ├── DataLoader
                │   └── LSTRBatchInferenceDataLayer.__getitem__()
                │       └── uniform_sampler()
                ├── LSTR.forward()
                ├── save .pkl scores
                └── compute_result['perframe']()
                    └── perframe_average_precision()
```

## 25. Full Stream Inference Call Tree

```text
tools/test_net.py
└── do_inference()
    └── do_lstr_batch_inference()
        └── do_lstr_stream_inference()
            ├── model.stream_inference()
            │   ├── feature_head_long.forward()
            │   ├── TransformerDecoder.stream_inference()
            │   │   └── TransformerDecoderLayer.stream_inference()
            │   │       └── MultiheadAttentionStream.stream_inference()
            │   │           └── DotProductAttentionStream.stream_inference()
            │   ├── feature_head_work.forward()
            │   ├── TransformerDecoder.forward()
            │   └── classifier
            └── compute_result['perframe']()
```

## 26. Tensor Flow Summary

For a typical two-stream LSTR training batch:

```text
Dataset returns:
visual: [B, L + W, visual_dim]
motion: [B, L + W, motion_dim]
mask:   [B, L]
target: [B, W, num_classes]

Feature head:
[B, T, visual_dim] + [B, T, motion_dim]
-> [B, T, visual_dim + motion_dim]
-> [B, T, d_model]

Transformer:
[B, T, d_model]
-> transpose
-> [T, B, d_model]

Model output:
[W, B, num_classes]
-> transpose
-> [B, W, num_classes]

Loss:
[B, W, C] -> [B * W, C]
```

Where:

- `L = LONG_MEMORY_NUM_SAMPLES`
- `W = WORK_MEMORY_NUM_SAMPLES`
- `C = DATA.NUM_CLASSES`

