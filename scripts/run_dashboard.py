"""
Run the minimal AgroEdge web dashboard (development / Pi LAN).

Environment:
  AGROEDGE_RUNTIME_LOG — path to JSONL log (default: logs/runtime_cycles.jsonl)
  FLASK_RUN_HOST — default 0.0.0.0
  FLASK_RUN_PORT — default 5000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_dashboard.app import create_app

if __name__ == "__main__":
    app = create_app()
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
