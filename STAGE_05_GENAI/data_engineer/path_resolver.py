"""
Path Resolver Utility for Stage 05 Data Engineering.
Dynamically locates HOSPITAL root and STAGE_05_GENAI root without hardcoded user paths.
Portable across any operating system, drive letter, and installation directory.
"""
from pathlib import Path
from typing import Optional, Union

def get_stage5_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """Locate the STAGE_05_GENAI directory dynamically by walking up parents."""
    if start_path is None:
        curr = Path(__file__).resolve()
    else:
        curr = Path(start_path).resolve()
        
    for p in [curr] + list(curr.parents):
        if p.name == "STAGE_05_GENAI":
            return p
        if (p / "STAGE_05_GENAI").is_dir():
            return p / "STAGE_05_GENAI"
    # Fallback to parent if inside data_engineer
    if curr.name == "data_engineer":
        return curr.parent
    return curr

def get_hospital_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """Locate the HOSPITAL workspace root directory dynamically."""
    stage5 = get_stage5_root(start_path)
    # The parent of STAGE_05_GENAI is the HOSPITAL workspace root
    if stage5.parent and (stage5.parent / "STAGE_05_GENAI").is_dir():
        return stage5.parent
        
    curr = Path(start_path).resolve() if start_path else Path(__file__).resolve()
    for p in [curr] + list(curr.parents):
        if (p / "stage1_ml").is_dir() or (p / "STAGE_01_ML").is_dir() or (p / "STAGE_02_DL").is_dir():
            return p
            
    return stage5.parent

def resolve_stage5_path(subpath: Union[str, Path]) -> Path:
    """Resolve a path relative to STAGE_05_GENAI."""
    p = Path(subpath)
    if p.is_absolute():
        return p
    return get_stage5_root() / p

def resolve_hospital_path(subpath: Union[str, Path]) -> Path:
    """Resolve a path relative to HOSPITAL workspace root."""
    p = Path(subpath)
    if p.is_absolute():
        return p
    return get_hospital_root() / p
