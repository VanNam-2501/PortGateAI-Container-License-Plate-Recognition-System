import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.getenv("APP_CONFIG_PATH", os.path.join("config", "settings.yaml"))
STATIC_DIR = os.path.join(PROJECT_ROOT, "app", "static")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")


def ensure_runtime_dirs() -> None:
    for path in (STATIC_DIR, DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR, RESULTS_DIR):
        os.makedirs(path, exist_ok=True)
