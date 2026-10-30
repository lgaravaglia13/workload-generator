"""Execution logging for the workload generator.

Progress logs (< WARNING) stream to stdout; warnings and errors stream to stderr.
"""
import logging
import sys

from models import describe_op

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"

ROOT_LOGGER_NAME = "workload"


def normalize_level(level_name: str) -> int:
    """Maps a case-insensitive level name to its logging constant."""
    level = logging.getLevelName(str(level_name).strip().upper())
    if not isinstance(level, int):
        valid = "DEBUG, INFO, WARNING, ERROR, CRITICAL"
        raise ValueError(f"unknown log level '{level_name}' (valid levels: {valid})")
    return level


def setup(level_name: str = "INFO") -> None:
    """Configures console logging handlers at the specified verbosity."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    progress = logging.StreamHandler(sys.stdout)
    progress.setLevel(logging.DEBUG)
    progress.addFilter(lambda record: record.levelno < logging.WARNING)
    progress.setFormatter(formatter)
    diagnostics = logging.StreamHandler(sys.stderr)
    diagnostics.setLevel(logging.WARNING)
    diagnostics.setFormatter(formatter)

    root.addHandler(progress)
    root.addHandler(diagnostics)
    root.setLevel(normalize_level(level_name))


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """Returns a logger namespaced under 'workload'."""
    if not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def attach_trace_progress(generator) -> None:
    """Subscribes the console progress narration listener to a TraceGenerator."""
    log = get_logger("trace")

    def narrate(context) -> None:
        if context.op_index == 0:
            log.info(
                "Phase %d/%d: %d operations",
                context.phase_position, context.total_phases, context.phase_op_count,
            )
        log.debug(
            "phase %s, op %d/%d: %s",
            context.phase_id, context.op_index + 1, context.phase_op_count,
            describe_op(context.op),
        )

    generator.add_op_listener(narrate)
