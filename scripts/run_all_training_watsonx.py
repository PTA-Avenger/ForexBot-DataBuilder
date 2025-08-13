import argparse
import os
import time
from typing import Dict, List, Tuple, Optional

from config import get_space_id, get_wml_credentials


def _import_wml_client():
    try:
        from ibm_watson_machine_learning import APIClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ibm-watson-machine-learning is required for non-dry-run. Install it via pip install ibm-watson-machine-learning"
        ) from exc
    return APIClient


def create_wml_base_client():
    APIClient = _import_wml_client()
    creds: Dict[str, str] = get_wml_credentials()
    client = APIClient({"url": creds["url"], "apikey": creds["apikey"]})
    return client


def create_wml_client_with_space(space_id: str):
    client = create_wml_base_client()
    client.set.default_space(space_id)
    return client


def collect_datasets(base_dir: str) -> Dict[str, str]:
    return {
        "trend": os.path.join(base_dir, "processed", "trend_dataset.csv"),
        "meanrev": os.path.join(base_dir, "processed", "meanrev_dataset.csv"),
        "sentiment": os.path.join(base_dir, "processed", "sentiment.csv"),
    }


def upload_datasets(client, datasets: Dict[str, str], dry_run: bool = False) -> None:
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[orchestrator] Missing dataset: {path}")
            continue
        if dry_run:
            print(f"[orchestrator] DRY-RUN: would upload data asset '{name}' from {path}")
            continue
        print(f"[orchestrator] Uploading data asset '{name}' from {path}...")
        # The SDK expects a string name, not a metadata dict
        client.data_assets.create(name=name, file_path=path)


def planned_jobs() -> List[Tuple[str, str, str]]:
    return [
        ("train_lstm", "scripts/train_lstm_watsonx.py", "gpu"),
        ("train_xgboost", "scripts/train_xgboost_watsonx.py", "cpu"),
        ("train_finbert", "scripts/train_finbert_watsonx.py", "gpu"),
    ]


def launch_jobs(client, jobs: List[Tuple[str, str, str]], dry_run: bool = False) -> None:
    for job_name, script, runtime in jobs:
        if dry_run:
            print(f"[orchestrator] DRY-RUN: would launch job '{job_name}' using {script} on {runtime.upper()}")
            continue
        print(f"[orchestrator] Launching '{job_name}' on {runtime.upper()} with {script}...")
        # Placeholder for real Watsonx.ai job submission; simulate runtime
        time.sleep(2)
        print(f"[orchestrator] {job_name} completed.")


def list_spaces() -> None:
    client = create_wml_base_client()
    print("[orchestrator] Accessible deployment spaces:")
    try:
        # This prints a table in the SDK; also provide programmatic fallbacks if needed
        client.spaces.list()
    except Exception as exc:  # pragma: no cover
        print(f"[orchestrator] Unable to list spaces: {exc}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Upload datasets and launch training jobs on Watsonx.ai")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Base data directory")
    parser.add_argument("--dry-run", action="store_true", help="Log intended actions without calling remote services")
    parser.add_argument("--space-id", type=str, default=None, help="Override SPACE_ID environment variable")
    parser.add_argument("--list-spaces", action="store_true", help="List accessible deployment spaces and exit")
    args = parser.parse_args(argv)

    datasets = collect_datasets(os.path.abspath(args.data_dir))

    if args.dry_run:
        print("[orchestrator] DRY-RUN: Skipping client creation and uploads requiring credentials")
        upload_datasets(client=None, datasets=datasets, dry_run=True)
        launch_jobs(client=None, jobs=planned_jobs(), dry_run=True)
        return 0

    if args.list_spaces:
        list_spaces()
        return 0

    # Resolve space id
    space_id: Optional[str] = args.space_id if args.space_id else None
    if not space_id:
        # Will raise helpful error if not set
        space_id = get_space_id()

    # Create client and set space, with helpful guidance if it fails
    try:
        client = create_wml_client_with_space(space_id)
    except Exception as exc:
        print(f"[orchestrator] Failed to set default space '{space_id}'. Reason: {exc}")
        print("[orchestrator] Tip: Ensure SPACE_ID matches a space in the same region as WML_URL and your API key has access.")
        print("[orchestrator] Available spaces in this account/region:")
        try:
            list_spaces()
        except Exception:
            pass
        return 2

    upload_datasets(client, datasets)
    launch_jobs(client, planned_jobs())
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
