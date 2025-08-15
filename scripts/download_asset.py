import argparse
import os
import sys
from typing import Optional, Dict

from config import get_space_id, get_wml_credentials


def _import_wml_client():
	try:
		from ibm_watson_machine_learning import APIClient  # type: ignore
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"ibm-watson-machine-learning is required. Install it via pip install ibm-watson-machine-learning"
		) from exc
	return APIClient


def _create_client():
	APIClient = _import_wml_client()
	creds: Dict[str, str] = get_wml_credentials()
	client = APIClient({"url": creds["url"], "apikey": creds["apikey"]})
	client.set.default_space(get_space_id())
	return client


def _resolve_asset_id_by_name(client, asset_name: str) -> Optional[str]:
	try:
		details = client.data_assets.get_details()
		resources = details.get("resources", [])
		for res in resources:
			meta = res.get("metadata", {})
			if meta.get("name") == asset_name:
				return meta.get("asset_id") or meta.get("id")
	except Exception:
		pass
	return None


def main(argv):
	parser = argparse.ArgumentParser(description="Download a data asset to a local path")
	parser.add_argument("--asset-id", type=str, default=None, help="Data asset ID to download")
	parser.add_argument("--asset-name", type=str, default=None, help="Data asset name to resolve and download")
	parser.add_argument("--output-path", type=str, required=True, help="Local file path to save the asset to")
	args = parser.parse_args(argv)

	if not args.asset_id and not args.asset_name:
		print("[download_asset] Provide either --asset-id or --asset-name", file=sys.stderr)
		return 2

	client = _create_client()

	asset_id = args.asset_id
	if not asset_id and args.asset_name:
		asset_id = _resolve_asset_id_by_name(client, args.asset_name)
		if not asset_id:
			print(f"[download_asset] Could not resolve asset id by name: {args.asset_name}", file=sys.stderr)
			return 3

	os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
	print(f"[download_asset] Downloading asset {asset_id} to {args.output_path} ...")
	client.data_assets.download(asset_id, filename=args.output_path)
	print("[download_asset] Done.")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))