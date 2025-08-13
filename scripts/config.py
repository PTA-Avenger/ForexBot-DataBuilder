import os
from typing import Dict

try:  # Make python-dotenv optional for dry-run scenarios
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv() -> None:  # type: ignore
        return None

# Load environment variables from a .env file if present
load_dotenv()


def get_wml_credentials() -> Dict[str, str]:
    """Return Watson Machine Learning credentials from environment.

    Raises an EnvironmentError only if one or more required variables are missing.
    This function is intentionally lazy so that importing this module never requires
    secrets to be present. Call it only right before making authenticated requests.
    """
    api_key = os.getenv("IBM_CLOUD_API_KEY")
    wml_url = os.getenv("WML_URL")
    missing = [name for name, value in {"IBM_CLOUD_API_KEY": api_key, "WML_URL": wml_url}.items() if not value]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: " + ", ".join(missing) + \
            ". Set them in your environment or a .env file before running authenticated operations."
        )
    return {"apikey": api_key, "url": wml_url}


def get_space_id() -> str:
    """Return the target Watsonx.ai Space ID.

    Raises an EnvironmentError if the SPACE_ID is not set.
    """
    space_id = os.getenv("SPACE_ID")
    if not space_id:
        raise EnvironmentError("Missing required environment variable: SPACE_ID. Set it before launching jobs.")
    return space_id
