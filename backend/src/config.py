import os
from typing import Optional
from google.oauth2 import service_account
from google.cloud import aiplatform

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_VERTEX_MODEL = os.getenv("GCP_VERTEX_MODEL", "gemini-1.5-flash")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", os.path.join(os.path.dirname(__file__), "..", "google-service.json"))


def init_vertex_ai(project: Optional[str] = None, location: Optional[str] = None):
    """Initialize Vertex AI SDK with optional explicit service account credentials."""
    project = project or GCP_PROJECT
    location = location or GCP_LOCATION
    if not project:
        # Try reading project_id from service account file
        if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
            try:
                import json
                with open(SERVICE_ACCOUNT_FILE, "r", encoding="utf-8") as f:
                    sa = json.load(f)
                    project = sa.get("project_id") or sa.get("project")
            except Exception:
                pass
    if not project:
        raise RuntimeError("GCP_PROJECT env var must be set or present in service account file")

    credentials = None
    if SERVICE_ACCOUNT_FILE and os.path.exists(SERVICE_ACCOUNT_FILE):
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

    aiplatform.init(project=project, location=location, credentials=credentials)

    return {
        "project": project,
        "location": location,
        "model": GCP_VERTEX_MODEL,
        "using_service_account": bool(credentials is not None),
        "aiplatform_version": getattr(aiplatform, "__version__", "unknown"),
    }
