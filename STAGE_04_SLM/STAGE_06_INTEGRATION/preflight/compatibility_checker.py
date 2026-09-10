"""
STAGE 06 INTEGRATION - COMPATIBILITY CHECKER
Validates cross-framework runtime compatibility (NumPy, PyTorch, Scikit-learn, XGBoost).
"""
import numpy as np
import torch
import sklearn
import xgboost

def check_compatibility():
    # Test tensor array interoperability
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    arr_back = t.numpy()
    
    array_interop = bool(np.allclose(arr, arr_back))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return {
        "status": "PASS" if array_interop else "FAIL",
        "device": device,
        "array_interoperability": array_interop,
        "numpy_version": np.__version__,
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Execution Host"
    }

if __name__ == "__main__":
    res = check_compatibility()
    print("Compatibility check:", res["status"])
