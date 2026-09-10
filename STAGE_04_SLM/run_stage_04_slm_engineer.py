"""
Convenience launcher for STAGE 04 SLM Engineer.
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
eng_runner = os.path.join(base_dir, "slm_engineer", "run_slm_engineer.py")
os.system(f"python {eng_runner}")
