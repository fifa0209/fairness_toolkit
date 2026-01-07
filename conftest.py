"""
Pytest Configuration (conftest.py)
Automatically runs before any test to fix import paths.
"""
import sys
from pathlib import Path

# Get the directory containing this file (Project Root)
project_root = Path(__file__).parent.resolve()

# Add project root to Python path so imports like 'measurement_module.src...' work
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"✅ [Pytest] Added to sys.path: {project_root}")