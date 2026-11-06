"""
Entrypoint to run the Fraud Detection API.

Usage:
    python run_api.py
    # or with custom host/port:
    FRAUD_THRESHOLD=0.6 python run_api.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

# Ensure the project root is importable when run from elsewhere
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("fraud-detection-api")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fraud Detection API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (overrides FRAUD_THRESHOLD env and metrics)",
    )
    args = parser.parse_args()

    if args.threshold is not None:
        import os

        os.environ["FRAUD_THRESHOLD"] = str(args.threshold)

    logger.info("Starting Fraud Detection API on %s:%d", args.host, args.port)
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
