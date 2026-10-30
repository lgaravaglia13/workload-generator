import argparse
from pathlib import Path
import sys


def load_workload_schedule(log, workload_arg: str) -> str:
    if workload_arg.endswith(".txt"):
        path = Path(workload_arg)
        if not path.is_file():
            log.error("workload file not found: %s", workload_arg)
            sys.exit(1)
        return path.read_text(encoding="utf-8").strip()
    return workload_arg


def run(args: argparse.Namespace) -> None:
    from dsl.parser import Lark_StandAlone
    from dsl.transformer import WorkloadTransformer
    from dataset_io import load_dataset
    from trace_generator import TraceGenerator
    from crash_dump import TraceCrashGuard, TraceCrashError
    from logger import attach_trace_progress, get_logger

    log = get_logger("cli")

    workload_schedule = load_workload_schedule(log, args.workload)

    lark_parser = Lark_StandAlone()
    transformer = WorkloadTransformer()

    log.info("Parsing workload: %s", args.workload)
    try:
        tree = lark_parser.parse(workload_schedule)
        workload = transformer.transform(tree)
        log.info("Parsed %d phases", len(workload.phases))
    except Exception as e:
        log.error("Parsing error: %s", e)
        sys.exit(1)

    if args.export_schedule:
        out_json_path = Path(args.export_schedule)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            f.write(workload.to_json(target_feature=args.key_col))
        log.info("Workload JSON saved to: %s", args.export_schedule)

    if not args.dataset:
        if args.export_schedule:
            return
        log.error("specify a dataset with -d/--dataset")
        sys.exit(1)

    log.info("Generating trace (seed=%s, rank_spread=%s)", args.seed, args.rank_spread)
    try:
        keys = load_dataset(
            args.dataset, key_col=args.key_col, dataset_format=args.dataset_format
        )
        generator = TraceGenerator(
            keys=keys,
            seed=args.seed,
            rank_spread=args.rank_spread,
        )
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)

    guard = TraceCrashGuard(generator, workload, dump_root=args.crash_dump_dir)
    attach_trace_progress(generator)

    try:
        total_ops = guard.run_to_csv(args.output_trace)
    except TraceCrashError as e:
        log.error("trace generation failed at %s: %s", e.location, e.error)
        if e.crash_dir is not None:
            log.error("crash report saved to: %s", e.crash_dir)
        else:
            log.error("failed to write crash report: %s", e.dump_error)
        sys.exit(1)
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)
    log.info("Trace generation complete: %s operations saved to %s", total_ops, args.output_trace)


def main():
    arg_parser = argparse.ArgumentParser(
        description=""
    )

    arg_parser.add_argument("-w", "--workload", help="Workload schedule DSL string, or path to a .txt file.")
    arg_parser.add_argument("-d", "--dataset", default=None, help="Path to the initial dataset (CSV or SOSD binary)")
    arg_parser.add_argument("-k", "--key-col", default="key", help="Key column name (default: 'key'; ignored by single-column formats)")
    arg_parser.add_argument(
        "-f",
        "--dataset-format",
        default="auto",
        choices=["auto", "csv", "sosd"],
        help="Input dataset format (default: 'auto', detected from the file)",
    )

    arg_parser.add_argument(
        "-e",
        "--export-schedule",
        nargs="?",
        const="./artifacts/schedule.json",
        default=None,
        metavar="PATH",
        help="Export parsed workload schedule to JSON (default: './artifacts/schedule.json' if specified without path)",
    )

    arg_parser.add_argument("-o", "--output-trace", default="./artifacts/workload_trace.csv", help="Output CSV path (default: 'workload_trace.csv')")
    arg_parser.add_argument(
        "--crash-dump-dir",
        default=None,
        metavar="PATH",
        help="Directory for crash reports (default: OS local data dir, "
        "e.g. '~/.local/share/workload-generator/crash-dumps' on Linux, "
        "'%%LOCALAPPDATA%%\\workload-generator\\crash-dumps' on Windows)",
    )
    arg_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    arg_parser.add_argument(
        "--rank-spread",
        type=float,
        default=0.025,
        help="Rank window radius as a fraction of n (default: 0.025, i.e. a 5%% window)",
    )
    arg_parser.add_argument(
        "--log-level",
        default="info",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: INFO)",
    )

    args = arg_parser.parse_args()

    from logger import setup, get_logger

    setup(args.log_level)
    log = get_logger("cli")

    if not args.workload:
        log.error("specify a workload with -w/--workload")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
