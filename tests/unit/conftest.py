# Pytest configuration for chamber_mpc unit tests
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PLUGIN_ROOT / "src"))
