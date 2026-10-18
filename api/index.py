"""
Vercel Serverless Function entrypoint for FastAPI backend.
Vercel looks for 'app' in api/index.py or api/*.py.
"""
import sys
import os

# Add root and backend to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(parent_dir, 'backend')

for path in [parent_dir, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.main import app
