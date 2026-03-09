# Autoresearch Skill

## What This Project Is

Autoresearch is an autonomous AI research framework. It gives an AI agent a small but real GPT-2 style language model training setup and lets the agent experiment autonomously: modify the code, train for 5 minutes, check if the result improved, keep or discard, and repeat indefinitely. The goal is to lower the **val_bpb** (validation bits per byte) metric through iterative experimentation.

## Project Structure

```
prepare.py      — constants, data prep, tokenizer, dataloader, evaluation (READ-ONLY)
train.py        — model, optimizer, training loop (THE ONLY FILE TO MODIFY)
program.md      — agent instructions and experiment workflow
pyproject.toml  — dependencies (do not add new ones)
CLAUDE.md       — project skill documentation
README.md       — project overview and setup instructions
```

## Key Commands

```bash
# One-time setup: download data + train tokenizer (~2 min)
uv run prepare.py

# Run a single training experiment (~5 min wall clock)
uv run train.py

# Redirect output for agent use (recommended)
uv run train.py > run.log 2>&1

# Extract results from a run
grep "^val_bpb:\|^peak_vram_mb:" run.log

# Check for crash output
tail -n 50 run.log
```

## Architecture Overview

### Data Flow

1. `prepare.py` downloads parquet shards from Hugging Face (`climbmix-400b-shuffle`), trains a BPE tokenizer with `rustbpe`, and wraps it in `tiktoken`
2. `train.py` imports `Tokenizer`, `make_dataloader`, and `evaluate_bpb` from `prepare.py`
3. Training runs for exactly 5 minutes (wall clock, excluding startup/compilation), then evaluates val_bpb
4. The agent records results, decides to keep (if val_bpb improved) or discard (if not), and iterates

### prepare.py — Fixed Infrastructure (DO NOT MODIFY)

- **Constants**: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300s`, `VOCAB_SIZE=8192`, `EVAL_TOKENS=40*524288`
- **Data**: Downloads parquet shards from HuggingFace, pinned validation shard `shard_06542.parquet`
- **Tokenizer**: BPE trained with `rustbpe`, wrapped in `tiktoken.Encoding`. Stored in `~/.cache/autoresearch/tokenizer/`
- **`make_dataloader()`**: Best-fit bin packing with 100% token utilization, zero padding. Pre-allocated pinned CPU/GPU buffers for async transfer. BOS-aligned rows.
- **`evaluate_bpb()`**: Vocab-independent bits-per-byte metric. Formula: `(sum(cross_entropy_nats) / log(2)) / sum(token_bytes)`. Excludes special tokens (byte length 0). Uses fixed `MAX_SEQ_LEN` for comparable results.

### train.py — Mutable Experiment File

**Model (`GPT` class)**:
- GPT-style transformer with configurable depth, width, heads
- `CausalSelfAttention`: Multi-query attention (GQA support), Flash Attention 3 via `kernels` package, rotary embeddings (RoPE), value embeddings (ResFormer) with input-dependent gating
- `MLP`: Squared ReLU activation (`relu(x)^2`), 4x expansion ratio
- `Block`: Pre-layer RMSNorm, residual connections
- Per-layer learnable `resid_lambdas` (init 1.0) and `x0_lambdas` (init 0.1) for blending with initial embedding
- Logit softcap: `15 * tanh(logits / 15)` (Gemini-style)
- Sliding window attention pattern: `SSSL` (S=half context, L=full context, last layer always L)
- Value embeddings on alternating layers (`has_ve` function)
- Model dimension calculated as `depth * ASPECT_RATIO`, rounded up to nearest `HEAD_DIM` multiple

**Optimizer (`MuonAdamW`)**:
- Hybrid optimizer: Muon for 2D matrix parameters (grouped by shape), AdamW for everything else
- Per-parameter-group learning rates: `EMBEDDING_LR=0.6`, `UNEMBEDDING_LR=0.004`, `MATRIX_LR=0.04`, `SCALAR_LR=0.5`
- AdamW LRs scaled by `1/sqrt(model_dim/768)` (µP-style scaling)
- Muon uses Polar Express orthogonalization (5 coefficients, configurable `ns_steps`) + NorMuon variance reduction + cautious weight decay
- Both `adamw_step_fused` and `muon_step_fused` are `torch.compile`-optimized
- Muon LR scaled by `sqrt(max(1, rows/cols))` for non-square matrices

**Training Loop**:
- Gradient accumulation: `TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)` steps
- Progress-based LR schedule: `progress = training_time / TIME_BUDGET`
- Warmdown: cosine decay in last 50% of training (`WARMDOWN_RATIO=0.5`)
- Muon momentum ramp: 0.85 → 0.95 over first 300 steps
- Weight decay schedule: `WEIGHT_DECAY * (1 - progress)` (linear decay)
- Fast fail: exits with code 1 if `train_loss > 100` (NaN/explosion detection)
- GC management: one full `gc.collect()` + `gc.freeze()` + `gc.disable()` after step 0, then periodic collect every 5000 steps
- First 10 steps are warmup (compilation), not counted toward `TIME_BUDGET`
- EMA smoothed training loss for logging (beta=0.9)

**Default Hyperparameters**:
- `DEPTH=8`, `ASPECT_RATIO=64` → model_dim=512, `HEAD_DIM=128` → 4 heads
- `DEVICE_BATCH_SIZE=128`, `TOTAL_BATCH_SIZE=2^19` (~524K tokens/step)
- `WEIGHT_DECAY=0.2`, `ADAM_BETAS=(0.8, 0.95)`, `WARMDOWN_RATIO=0.5`
- `WARMUP_RATIO=0.0`, `FINAL_LR_FRAC=0.0`, `WINDOW_PATTERN="SSSL"`

**Weight Initialization**:
- Embedding: `normal_(mean=0, std=1.0)`
- Unembedding (lm_head): `normal_(mean=0, std=0.001)`
- Q/K/V projections: `uniform_(-s, s)` where `s = sqrt(3) * n_embd^(-0.5)`
- Output projections (attn, MLP): `zeros_()`
- Residual lambdas: `fill_(1.0)`, x0 lambdas: `fill_(0.1)`
- Value embeddings: `uniform_(-s, s)`, VE gate weights: `zeros_()` (sigmoid(0)=0.5, ×2 = neutral 1.0)
- Embeddings and value embeddings cast to bf16 after init

## Experiment Workflow (from program.md)

### Setup Phase
1. Agree on a run tag, create branch `autoresearch/<tag>`
2. Read `README.md`, `prepare.py`, `train.py`
3. Verify `~/.cache/autoresearch/` has data shards and tokenizer
4. Initialize `results.tsv` with header: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`
5. Run baseline (unmodified `train.py`) as first experiment

### Experiment Loop (runs forever)
1. Modify `train.py` with an experimental idea
2. `git commit -am "description"`
3. `uv run train.py > run.log 2>&1`
4. Extract results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. If grep empty → crash: check `tail -n 50 run.log`
6. Record in `results.tsv`
7. If val_bpb improved → keep commit
8. If val_bpb worse/equal → `git reset --hard HEAD~1`
9. If timeout > 10 min → kill and treat as crash
10. **Never stop** — loop until manually interrupted

### Constraints
- **CAN**: Modify `train.py` (architecture, optimizer, hyperparameters, batch size, model size, delete code)
- **CANNOT**: Modify `prepare.py`, add dependencies, change evaluation metric, change time budget

### Simplicity Criterion
All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

### Results Format (results.tsv, tab-separated)
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	0.000000	0.0	crash	double model width (OOM)
```

## Key Metric: val_bpb

- **Lower is better**
- Bits per byte — vocab-size-independent, fair across architecture changes
- Formula: cross-entropy in nats, converted to bits, divided by total byte count of target tokens
- Evaluated on pinned validation shard (shard 06542)

## Output Format

After each training run, the script prints:
```
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

## Conventions

- Python 3.10+, managed with `uv`
- Single NVIDIA GPU (tested on H100), CUDA 12.8
- `torch.compile` used extensively for performance (model compiled with `dynamic=False`)
- `bfloat16` autocast during training
- Deterministic seeding: `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`
- No automated tests — the val_bpb metric is the test
- Simplicity criterion: simpler code that matches performance is preferred over complex code with marginal gains
- No CLI flags for hyperparameters — edit constants directly in `train.py`
- Flash Attention 3: uses `varunneal/flash-attention-3` on Hopper (sm90), `kernels-community/flash-attn3` on other GPUs

## Dependencies (from pyproject.toml)

- `torch==2.9.1` (CUDA 12.8 wheel)
- `kernels>=0.11.7` (Flash Attention 3)
- `numpy>=2.2.6`, `pandas>=2.3.3`, `pyarrow>=21.0.0`
- `requests>=2.32.0` (data download)
- `rustbpe>=0.1.0`, `tiktoken>=0.11.0` (tokenizer)
- `matplotlib>=3.10.8` (analysis/plotting)
