"""
Tests for Stage 05 EDA Subsystem.
"""
import pytest
from pathlib import Path
from eda_prompteng.eda import OncologyEDA

def test_eda_generation():
    eda = OncologyEDA()
    summary = eda.run_eda()
    assert "dataset_dimensions" in summary
    assert summary["dataset_dimensions"]["rows"] > 0
    assert summary["dataset_dimensions"]["columns"] > 0

    rep_dir = Path("STAGE_05_GENAI/outputs/reports")
    assert (rep_dir / "eda_summary.json").exists()
    assert (rep_dir / "eda_report.md").exists()
    assert (rep_dir / "eda_distributions.png").exists()
