"""
STAGE 06 INTEGRATION - MASTER INTEGRATION PREFLIGHT
Runs dependency, artifact, and compatibility checks, recording preflight_report.json.
"""
import os
import json
import psutil
from STAGE_06_INTEGRATION.preflight.dependency_checker import check_dependencies
from STAGE_06_INTEGRATION.preflight.artifact_checker import check_artifacts
from STAGE_06_INTEGRATION.preflight.compatibility_checker import check_compatibility

def run_preflight():
    print("=" * 65)
    print("STAGE 06 INTEGRATION - PHASE 0: PREFLIGHT VERIFICATION")
    print("=" * 65)
    
    dep_res = check_dependencies()
    art_res = check_artifacts()
    com_res = check_compatibility()
    
    mem = psutil.virtual_memory()
    overall_status = "PASS" if (dep_res["status"] == "PASS" and art_res["status"] == "PASS" and com_res["status"] == "PASS") else "FAIL"
    
    report = {
        "stage": "STAGE_06_INTEGRATION",
        "preflight_status": overall_status,
        "dependencies": dep_res,
        "artifacts": art_res,
        "compatibility": com_res,
        "hardware": {
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_available_gb": round(mem.available / (1024**3), 2)
        }
    }
    
    curr = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(curr)
    
    out_path = os.path.join(base, "outputs", "preflight_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"Preflight status: {overall_status}")
    print(f"Preflight report written to: {out_path}")
    return report

if __name__ == "__main__":
    run_preflight()
