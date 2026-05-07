#!/usr/bin/env python3
"""Set up a development environment for chamber-mpc.

Cross-platform (Linux, macOS, Windows). Creates a virtual environment
and installs test dependencies.

Usage:
    python setup_dev_env.py              # create dev-env and install deps
    python setup_dev_env.py --run-tests  # also run the test suite
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
VENV_DIR = REPO_ROOT / "dev-env"
REQUIREMENTS = ["pytest>=7.0", "pytest-cov>=4.0"]


def main():
    run_tests = "--run-tests" in sys.argv

    print("chamber-mpc development environment setup")
    print("=" * 50)
    print("Repository root: %s" % REPO_ROOT)
    print("Python version:  %s" % sys.version.split()[0])
    print("Platform:        %s" % sys.platform)
    print("Venv location:   %s" % VENV_DIR)
    print()

    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ is required. Current: %d.%d"
              % sys.version_info[:2])
        sys.exit(1)

    if VENV_DIR.exists():
        answer = input("Virtual environment already exists. Recreate? [y/N] ")
        if answer.strip().lower() == "y":
            import shutil
            shutil.rmtree(VENV_DIR)
        else:
            print("Keeping existing venv.")

    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        venv.create(str(VENV_DIR), with_pip=True)
        print("  Created.")

    if sys.platform == "win32":
        pip_bin = VENV_DIR / "Scripts" / "pip"
        python_bin = VENV_DIR / "Scripts" / "python"
    else:
        pip_bin = VENV_DIR / "bin" / "pip"
        python_bin = VENV_DIR / "bin" / "python"

    print("Installing dependencies: %s" % ", ".join(REQUIREMENTS))
    subprocess.check_call(
        [str(pip_bin), "install", "--quiet"] + REQUIREMENTS)
    print("  Installed.")

    print("Verifying chamber_mpc import...")
    result = subprocess.run(
        [str(python_bin), "-c",
         "import sys; sys.path.insert(0, 'src'); "
         "from chamber_mpc.h_interpolator import HInterpolator; "
         "h = HInterpolator([(100, 0.15)]); "
         "print('  OK: h(100) = %.4f' % h.h(100))"],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True)
    if result.returncode != 0:
        print("  FAILED: %s" % result.stderr.strip())
        sys.exit(1)
    print(result.stdout.strip())

    if run_tests:
        print()
        print("Running tests...")
        print("-" * 50)
        result = subprocess.run(
            [str(python_bin), "-m", "pytest", "tests/unit/", "-v",
             "--cov", "--cov-report=term-missing"],
            cwd=str(REPO_ROOT))
        sys.exit(result.returncode)

    print()
    print("=" * 50)
    print("Setup complete! To run tests:")
    print()
    if sys.platform == "win32":
        print("  dev-env\\Scripts\\python -m pytest tests\\unit\\ -v")
    else:
        print("  dev-env/bin/python -m pytest tests/unit/ -v")
    print()


if __name__ == "__main__":
    main()
