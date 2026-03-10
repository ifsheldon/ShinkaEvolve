import json
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


def load_configs_from_yaml(config_path: str):
    """
    Loads configs from a YAML file.
    """
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    assert "db_config" in configs, "db_config not found in config file"
    assert "evo_config" in configs, "evo_config not found in config file"

    db_cfg = DatabaseConfig(**configs["db_config"])
    evo_cfg = EvolutionConfig(**configs["evo_config"])
    return evo_cfg, db_cfg


def load_results(results_dir: str):
    """
    Loads results from the specified directory.

    Args:
        results_dir: The directory containing the results.

    Returns:
        dict: A dictionary containing the loaded results.
    """
    loaded_results = {"correct": {"correct": False}, "metrics": {}}
    results_dir_path = Path(results_dir)

    stdout_log_path = results_dir_path / "job_log.out"
    if stdout_log_path.exists():
        with open(stdout_log_path, "r") as f:
            loaded_results["stdout_log"] = f.read()
    else:
        loaded_results["stdout_log"] = ""

    stderr_log_path = results_dir_path / "job_log.err"
    if stderr_log_path.exists():
        with open(stderr_log_path, "r") as f:
            loaded_results["stderr_log"] = f.read()
    else:
        loaded_results["stderr_log"] = ""

    metrics_file_path = results_dir_path / "metrics.json"
    if metrics_file_path.exists():
        with open(metrics_file_path, "r") as f:
            try:
                loaded_results["metrics"] = json.load(f)
            except json.JSONDecodeError:
                file_path_str = str(metrics_file_path)
                warning_msg = f"Could not decode JSON from {file_path_str}"
                logger.warning(warning_msg)
                loaded_results["metrics"] = {}
    else:
        file_path_str = str(metrics_file_path)
        warning_msg = f"Metrics file not found at {file_path_str}"
        logger.warning(warning_msg)
        loaded_results["metrics"] = {}

    correct_file_path = results_dir_path / "correct.json"
    if correct_file_path.exists():
        with open(correct_file_path, "r") as f:
            loaded_results["correct"] = json.load(f)
        # Infer error_type for backward compatibility
        if not loaded_results["correct"].get("correct", False):
            if "error_type" not in loaded_results["correct"]:
                loaded_results["correct"]["error_type"] = "runtime_error"
    else:
        loaded_results["correct"] = {
            "correct": False,
            "error_type": "crash",
            "error": "Process terminated without writing results",
        }

    return loaded_results


def parse_time_to_seconds(time_str: str) -> int:
    """Converts hh:mm:ss to seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Time format must be hh:mm:ss")
    h, m, s = [int(p) for p in parts]
    return h * 3600 + m * 60 + s


def write_timeout_marker(results_dir: str, timeout_seconds: int = None):
    """Write a correct.json marking this evaluation as timed out.

    Called by the scheduler/monitor after killing a timed-out process,
    before load_results() is called.  Also overwrites metrics.json so
    that any partially-written scores from the killed process are zeroed
    out.
    """
    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    correct_data = {
        "correct": False,
        "error": (
            f"Evaluation timed out after {timeout_seconds}s"
            if timeout_seconds
            else "Evaluation timed out"
        ),
        "error_type": "timeout",
    }
    with open(results_dir_path / "correct.json", "w") as f:
        json.dump(correct_data, f)

    # Zero out combined_score in any metrics the evaluation may have
    # written before being killed, preserving all other data.
    metrics_path = results_dir_path / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics["combined_score"] = 0.0
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
        except (json.JSONDecodeError, OSError):
            pass
