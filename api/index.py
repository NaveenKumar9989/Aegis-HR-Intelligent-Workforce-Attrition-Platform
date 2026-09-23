"""
Vercel Serverless Function Entry Point for Aegis HR Flask Application.
Exposes WSGI `app` callable for Vercel's Python runtime.
"""

import sys
from pathlib import Path

# Add project root to sys.path so that 'backend' and all submodules can be imported cleanly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app import app

# Vercel's Python builder automatically inspects and binds to `app`
