"""
STAGE 06 INTEGRATION - DEPENDENCY CHECKER
Validates that all required packages and runtime environments are installed.
"""
import sys
import psutil
import torch
import transformers
import peft
import sklearn
import xgboost

def check_dependencies():
    packages = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "psutil": psutil.__version__
    }
    
    # Validation gates
    ok = (
        sys.version_info >= (3, 10) and
        hasattr(torch, "__version__") and
        hasattr(transformers, "__version__") and
        hasattr(peft, "__version__")
    )
    return {"status": "PASS" if ok else "FAIL", "packages": packages}

if __name__ == "__main__":
    res = check_dependencies()
    print("Dependencies check:", res["status"])
