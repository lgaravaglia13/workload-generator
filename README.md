This project is a workload generator for benchmarking learned indexes under controlled data drift.

The workload is composed by a sequence of phases described by a domain specific language.
Each operation specifies the distribution and a target percentile so that reads, inserts, updates and deletes
can be targeted towards a specific region of the dataset. The workload generator outputs a sequence of primitive index operations that allows
reproducing the intended dataset evolution.

This generator follows a fail-fast policy in order to prevents undeclared distribution degradation.
For example, if the target region for an insert doesn't contain enough free key capacity, it throws a runtime error rather than widening
the window. The same applies when the interval is smaller than the requested batch size. 

## Trade-offs
- When a gap within the targeted rank window doesn't have enough free space to match the requested distribution, excess quota is redistributed over neighboring gaps in the same window.
- Inserted keys are normally drawn uniformly from selected gaps. However, in large outlier gaps, keys are placed near existing records (bounded by the local gap scale) rather than scattered across the entire gap space.
- **Drift Monitoring:** Allocation drift can be tracked at runtime via the logger (`DEBUG` level), which reports exact unhostable mass and quota spill per batch.

## Setup
To install the required dependencies run
```bash
pip install -r requirements.txt
```

Then you can run it:
```bash
python3 main.py \
  --workload workload-examples/default.txt \
  --dataset "path-to-initial-dataset" \
  --output-trace "workload_trace.csv"
```

## Command Line Options

### Required

| Option | Description |
|--------|-------------|
| `-w`, `--workload` `TEXT` | Workload schedule: inline DSL string or path to a `.txt` file (see [Workload DSL](docs/WORKLOAD.md)) |
| `-d`, `--dataset` `PATH` | Path to the initial dataset (CSV or SOSD binary) |

### Trace options

| Option | Description |
|--------|-------------|
| `-o`, `--output-trace` `PATH` | Output trace path (default: `./artifacts/workload_trace.csv`); the format is chosen by extension (`.csv`) |
| `--seed` `N` | Random seed for reproducibility (default: `42`) |
| `--rank-spread` `N` | Rank window radius as a fraction of the dataset size (default: `0.025`, i.e. a 5% window): widens or tightens the hotspot every targeted operation samples from |

### Dataset options

| Option | Description |
|--------|-------------|
| `-k`, `--key-col` `NAME` | Key column name for CSV datasets (default: `key`; ignored by single-column formats) |
| `-f`, `--dataset-format` `{auto,csv,sosd}` | Input dataset format (default: `auto`, detected from the file. See [Input dataset formats](#input-dataset-formats)) |

### Schedule options

| Option | Description |
|--------|-------------|
| `-e`, `--export-schedule` `[PATH]` | Dump the parsed schedule as JSON without generating a trace (default path: `./artifacts/schedule.json` when the flag is given without a value) |

### Logging & diagnostics

| Option | Description |
|--------|-------------|
| `--log-level` `{DEBUG,INFO,WARNING,ERROR}` | Console log verbosity (default: `INFO`. See [Logging](#logging)) |
| `--crash-dump-dir` `PATH` | Directory for crash reports (default: OS local data dir, e.g. `~/.local/share/workload-generator/crash-dumps` on Linux, `%LOCALAPPDATA%\workload-generator\crash-dumps` on Windows) |

### Input dataset formats
The `--dataset` flag accepts multiple on-disk formats. The format is
auto-detected unless you force one with `--dataset-format`:

| Format | `--dataset-format` | Description |
|---|---|---|
| CSV | `csv` | One column per key (`--key-col` selects it, default `key`). |
| SOSD binary | `sosd` | One `uint64` size header followed by that many `uint64` keys. |

### Logging
Console output goes through a logger, controlled by `--log-level`
(default `INFO`):

| Level | Verbosity |
|---|---|
| `DEBUG` | Everything below, plus per-operation detail and the `sampler.drift` insert-allocation diagnostics. |
| `INFO` | Pipeline narration: parsing, dataset loading, phase progress, write milestones, final summary. |
| `WARNING` | Recoverable anomalies only (e.g. an unordered dataset being sorted in memory). |
| `ERROR` | Terminal failures. |

Progress (`< WARNING`) is written to stdout, warnings and errors to stderr,
so failures stay visible when stdout is redirected.

### Dev
Make sure to install the dev dependencies:
```bash
pip install -r requirements-dev.txt
```

- If you modify `dsl/grammar.lark`, regenerate the parser by running
```bash
./dsl/regenerate_parser.sh
```

- Execute test suite:
```bash
python3 -m unittest discover -s tests
```

## More details
- [Workload DSL](docs/WORKLOAD.md).
- [Sampling](docs/SAMPLING.md).
