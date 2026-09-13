"""
Stage 02 DL Adapter for Stage 05.
Adapts pathology images to Stage 02 EfficientNet-B0 vision classifier.
STRICT GOVERNANCE RULE:
If a verified pathology image file exists -> execute Stage 02 inference.
Else -> mark strictly as NOT_APPLICABLE. Never fabricate fake pathology images or results.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Windows OpenMP & Torch DLL safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")
STAGE06_ROOT = HOSPITAL_ROOT / "STAGE_04_SLM" / "STAGE_06_INTEGRATION"

if str(HOSPITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HOSPITAL_ROOT))
if str(STAGE06_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE06_ROOT))

class Stage2Adapter:
    def __init__(self):
        self._dl_adapter = None

    def _ensure_dl_adapter(self):
        if self._dl_adapter is None:
            try:
                from adapters.dl_adapter import DLAdapter
                self._dl_adapter = DLAdapter()
            except Exception:
                try:
                    from STAGE_06_INTEGRATION.adapters.dl_adapter import DLAdapter
                    self._dl_adapter = DLAdapter()
                except Exception as e:
                    print(f"[Stage2Adapter] Notice: Upstream DLAdapter unavailable: {e}")

    def predict(self, pathology_specimen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes genuine Stage 02 EfficientNet-B0 inference ONLY if a verified image exists.
        Otherwise returns NOT_APPLICABLE immediately without importing or fabricating.
        """
        if not pathology_specimen:
            return {
                "status": "NOT_APPLICABLE",
                "reason": "No digital histopathology specimen provided for this synthetic scenario profile.",
                "predicted_class": None,
                "class_label": None,
                "confidence": None,
                "feature_embedding": None
            }

        image_path = pathology_specimen.get("tile_path")
        has_image = pathology_specimen.get("has_image", False)

        if not has_image or not image_path or not os.path.exists(str(image_path)):
            return {
                "status": "NOT_APPLICABLE",
                "reason": f"Pathology image path '{image_path}' does not exist on disk. Image inference skipped under strict governance.",
                "predicted_class": None,
                "class_label": None,
                "confidence": None,
                "feature_embedding": None
            }

        # Lazy initialize DLAdapter only when a valid physical image actually exists on disk
        self._ensure_dl_adapter()

        if self._dl_adapter is not None:
            try:
                res = self._dl_adapter.predict(str(image_path))
                return {
                    "status": "SUCCESS",
                    "predicted_class": res.get("predicted_class"),
                    "class_label": res.get("class_label"),
                    "confidence": res.get("confidence"),
                    "class_probabilities": res.get("class_probabilities"),
                    "feature_embedding": res.get("feature_embedding")[:10] if res.get("feature_embedding") else None,
                    "model_used": "Stage 02 EfficientNet-B0 CNN"
                }
            except Exception as e:
                return {
                    "status": "ERROR",
                    "reason": f"Inference error on valid image: {str(e)}"
                }

        return {
            "status": "NOT_APPLICABLE",
            "reason": "Stage 02 DL model weight engine not initialized."
        }

if __name__ == "__main__":
    s2 = Stage2Adapter()
    print("Empty image test:", s2.predict(None))
