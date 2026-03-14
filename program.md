# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Verify wandb**: Ensure `WANDB_API_KEY` is set in the environment, or run `wandb login`. If wandb is not configured, training will still work normally without logging.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies beyond what's in `pyproject.toml`. (`wandb` is already included.)
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
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

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Querying Past Experiments (wandb)

All experiments are automatically logged to wandb (project: `autoresearch`). Before deciding what to try next, query wandb for insights. Run this Python snippet:

```python
import wandb
api = wandb.Api()
runs = api.runs("autoresearch", order="-summary_metrics.val_bpb")
print(f"{'val_bpb':>10} {'DEPTH':>5} {'MATRIX_LR':>10} {'commit':>8}  name")
for run in runs:
    s = run.summary
    c = run.config
    val = s.get("val_bpb", "N/A")
    depth = c.get("DEPTH", "?")
    mlr = c.get("MATRIX_LR", "?")
    commit = c.get("git_commit", "?")
    print(f"{val:>10} {depth:>5} {mlr:>10} {commit:>8}  {run.name}")
```

Use this to:
- See which hyperparameter ranges have been tried
- Identify which changes improved val_bpb
- Avoid re-running experiments that already failed
- Spot trends (e.g., "higher MATRIX_LR helps", "DEPTH > 10 hurts at this budget")

You can also inspect individual runs:
```python
for run in runs:
    print(run.config)  # all hyperparameters
    print(run.summary) # all final metrics
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Query wandb for past experiment insights (every few iterations, not necessarily every single one) to inform your next hypothesis
3. Tune `train.py` with an experimental idea by directly hacking the code.
4. git commit
5. **Run the experiment in the background** and monitor it:
   ```bash
   uv run train.py > run.log 2>&1 &
   TRAIN_PID=$!
   ```
6. **Monitor training** — check progress every 60 seconds:
   ```bash
   grep "^CHECKPOINT" run.log | tail -3
   ```
   The CHECKPOINT lines have the format: `CHECKPOINT step=N loss=X.XXXXXX progress=P mfu=M`
7. **Early termination** — kill the run early if it looks bad (saves time for more experiments):
   - **Loss diverging**: if loss is increasing or much higher than your best baseline's loss at similar progress, kill it:
     ```bash
     kill $TRAIN_PID 2>/dev/null; wait $TRAIN_PID 2>/dev/null
     ```
   - **Crash/stuck**: if no new CHECKPOINT lines appear after 2+ minutes, the run is stuck.
   - After killing, treat it as a `discard` or `crash` and revert.
   - **When NOT to kill early**: the first ~50 steps are warmup/compilation — don't judge loss there. Wait for at least 2-3 CHECKPOINT lines before deciding.
8. **If the run completes normally**, read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
9. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
10. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
11. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
12. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Early kill strategy**: By monitoring loss during training, you can kill clearly-failing experiments after 1-2 minutes instead of waiting the full 5 minutes. This means you can run ~20-30 experiments per hour instead of ~12. Use your best judgment: don't be too aggressive (some experiments need time to converge), but don't waste time on obvious failures (loss 2x worse than baseline, loss going up instead of down).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. With early termination of bad experiments, you can run 20-30 experiments per hour (instead of ~12 if you always wait the full 5 minutes). Over 8 hours of sleep, that's 160-240 experiments! The user then wakes up to experimental results, all completed by you while they slept!
