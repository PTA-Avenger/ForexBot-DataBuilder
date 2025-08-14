import argparse
import os
import time
import tempfile
import zipfile
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


def upload_datasets(client, datasets: Dict[str, str], dry_run: bool = False) -> Dict[str, Optional[str]]:
    uploaded: Dict[str, Optional[str]] = {}
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[orchestrator] Missing dataset: {path}")
            uploaded[name] = None
            continue
        if dry_run:
            print(f"[orchestrator] DRY-RUN: would upload data asset '{name}' from {path}")
            uploaded[name] = None
            continue
        print(f"[orchestrator] Uploading data asset '{name}' from {path}...")
        details = client.data_assets.create(name=name, file_path=path)
        try:
            asset_id = details.get("metadata", {}).get("asset_id") or details.get("asset_id")
        except Exception:
            asset_id = None
        uploaded[name] = asset_id
    return uploaded


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
        client.spaces.list()
    except Exception as exc:  # pragma: no cover
        print(f"[orchestrator] Unable to list spaces: {exc}")


def _zip_code(script_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="wml_pkg_")
    zip_path = os.path.join(tmp_dir, os.path.splitext(os.path.basename(script_path))[0] + ".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Put the script at the root of the package
        zf.write(script_path, arcname=os.path.basename(script_path))
    return zip_path


def _resolve_software_spec_id(client, preferred_names: List[str]) -> str:
    for name in preferred_names:
        try:
            sw_id = client.software_specifications.get_id_by_name(name)
            if sw_id:
                return sw_id
        except Exception:
            continue
    raise RuntimeError(
        "Could not resolve a software specification id. Specify one explicitly with --software-spec-name."
    )


def submit_training_job(client, job_name: str, script_path: str, software_spec_name: str, hardware_name: str, hardware_nodes: int, wait: bool = False) -> Optional[str]:
    # Prefer training_definitions if available, otherwise fall back to repository API
    code_zip = _zip_code(script_path)
    command = f"python {os.path.basename(script_path)}"

    sw_id = _resolve_software_spec_id(client, [software_spec_name, "default_py3.10", "runtime-22.2-py3.10", "python-3.10"])

    if hasattr(client, "training_definitions"):
        td = client.training_definitions
        tr = client.training
        meta = {
            td.ConfigurationMetaNames.NAME: job_name,
            td.ConfigurationMetaNames.DESCRIPTION: f"Training job for {job_name}",
            td.ConfigurationMetaNames.SOFTWARE_SPEC_UID: sw_id,
            td.ConfigurationMetaNames.HARDWARE_SPEC: {"name": hardware_name, "nodes": int(hardware_nodes)},
            td.ConfigurationMetaNames.COMMAND: command,
        }
        print(f"[orchestrator] Creating training definition for {job_name} (software_spec={software_spec_name}, hardware={hardware_name} x{hardware_nodes})")
        td_details = td.store(meta_props=meta, training_definition=code_zip)
        try:
            td_id = td.get_id(td_details)
        except Exception:
            td_id = td_details.get("metadata", {}).get("id")  # type: ignore[attr-defined]
        print(f"[orchestrator] Submitting training run for {job_name}...")
        run_details = tr.run(training_definition_id=td_id)
        try:
            job_id = tr.get_id(run_details)
        except Exception:
            job_id = run_details.get("metadata", {}).get("id")  # type: ignore[attr-defined]
    else:
        # Repository fallback (compatible with multiple ibm-watson-machine-learning SDK variants)
        repo = client.repository
        tr = client.training

        # Try to find a MetaNames class that matches training definitions across SDK versions
        repo_meta_names = None
        for candidate_attr in ("DefinitionMetaNames", "TrainingDefinitionMetaNames"):
            if hasattr(repo, candidate_attr):
                repo_meta_names = getattr(repo, candidate_attr)
                break
        if repo_meta_names is None and hasattr(tr, "DefinitionMetaNames"):
            # Some versions expose it under training
            repo_meta_names = getattr(tr, "DefinitionMetaNames")

        def resolve_meta_key(attribute_name: str, default_key: str) -> str:
            try:
                if repo_meta_names is not None and hasattr(repo_meta_names, attribute_name):
                    return getattr(repo_meta_names, attribute_name)
            except Exception:
                pass
            return default_key

        meta = {
            resolve_meta_key("NAME", "name"): job_name,
            resolve_meta_key("DESCRIPTION", "description"): f"Training job for {job_name}",
            resolve_meta_key("SOFTWARE_SPEC_UID", "software_spec_uid"): sw_id,
            resolve_meta_key("HARDWARE_SPEC", "hardware_spec"): {"name": hardware_name, "nodes": int(hardware_nodes)},
            resolve_meta_key("COMMAND", "command"): command,
        }
        print(f"[orchestrator] Creating training definition (repository) for {job_name} (software_spec={software_spec_name}, hardware={hardware_name} x{hardware_nodes})")
        # Try multiple repository store methods/args for cross-version compatibility
        td_details = None
        last_exc = None
        for method_name in ("store_training_definition", "store_definition", "store"):
            store_method = getattr(repo, method_name, None)
            if store_method is None:
                continue
            # Try with parameter name training_definition
            try:
                td_details = store_method(training_definition=code_zip, meta_props=meta)
                break
            except Exception as exc:
                last_exc = exc
                # Try alternate parameter name: definition
                try:
                    td_details = store_method(definition=code_zip, meta_props=meta)  # type: ignore[call-arg]
                    break
                except Exception as exc2:  # noqa: F841
                    last_exc = exc2
                    continue
        if td_details is None:
            raise RuntimeError(f"Unable to store training definition in repository: {last_exc}")
        try:
            td_id = repo.get_definition_uid(td_details)
        except Exception:
            try:
                td_id = repo.get_definition_id(td_details)  # type: ignore[attr-defined]
            except Exception:
                td_id = td_details.get("metadata", {}).get("id")  # type: ignore[attr-defined]
        print(f"[orchestrator] Submitting training run for {job_name}...")
        # Handle SDKs expecting different parameter names
        try:
            run_details = tr.run(training_definition_id=td_id)
        except TypeError:
            run_details = tr.run(training_definition_uid=td_id)
        try:
            job_id = tr.get_id(run_details)
        except Exception:
            job_id = run_details.get("metadata", {}).get("id")  # type: ignore[attr-defined]

    print(f"[orchestrator] Submitted {job_name}, job_id={job_id}")

    if wait:
        while True:
            try:
                status = tr.get_status(job_id)
                state = status.get("state") or status.get("status")
                print(f"[orchestrator] {job_name} status: {state}")
                if str(state).lower() in {"completed", "failed", "canceled", "error"}:
                    break
            except Exception as exc:
                print(f"[orchestrator] Error fetching status: {exc}")
                break
            time.sleep(15)

    return job_id


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Upload datasets and launch training jobs on Watsonx.ai")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Base data directory")
    parser.add_argument("--dry-run", action="store_true", help="Log intended actions without calling remote services")
    parser.add_argument("--space-id", type=str, default=None, help="Override SPACE_ID environment variable")
    parser.add_argument("--list-spaces", action="store_true", help="List accessible deployment spaces and exit")
    parser.add_argument("--real", action="store_true", help="Submit real training jobs instead of simulated launches")
    parser.add_argument("--software-spec-name", type=str, default="runtime-24.1-py3.11", help="Software spec name to use for jobs")
    parser.add_argument("--hardware-name-cpu", type=str, default="S", help="Hardware spec name for CPU jobs")
    parser.add_argument("--hardware-name-gpu", type=str, default="S", help="Hardware spec name for GPU jobs")
    parser.add_argument("--hardware-nodes", type=int, default=1, help="Number of nodes for each job")
    parser.add_argument("--wait", action="store_true", help="Wait for submitted jobs to complete")
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
        space_id = get_space_id()

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

    uploaded_asset_ids = upload_datasets(client, datasets)

    if args.real:
        jobs = planned_jobs()
        for job_name, script, runtime in jobs:
            hardware_name = args.hardware_name_gpu if runtime == "gpu" else args.hardware_name_cpu
            try:
                submit_training_job(
                    client=client,
                    job_name=job_name,
                    script_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", script)),
                    software_spec_name=args.software_spec_name,
                    hardware_name=hardware_name,
                    hardware_nodes=args.hardware_nodes,
                    wait=args.wait,
                )
            except Exception as exc:
                print(f"[orchestrator] Failed to submit job {job_name}: {exc}")
        return 0

    # Default simulated launch if --real not provided
    launch_jobs(client, planned_jobs())
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
