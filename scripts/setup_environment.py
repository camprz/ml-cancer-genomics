import sys
from pathlib import Path

def setup_paths():
    """
    Sets up the paths for the project by adding the scripts directory
    to the Python path and returning the project root.
    """
    # Always find the root relative to this script's location
    project_root = Path(__file__).resolve().parent.parent

    # Add the scripts directory to the Python path
    sys.path.append(str(project_root / "scripts"))

    return project_root
