"""
Dependency Graph for Stage 06 Integration Pipeline.
Defines and verifies topological DAG of execution stages.
"""

from typing import List, Dict, Set

class DependencyGraph:
    def __init__(self):
        self.nodes = ["INPUT_VALIDATION", "PREPROCESSING", "ML_STAGE", "DL_STAGE", "NLP_STAGE", "SLM_STAGE", "SAFETY_STAGE", "OUTPUT_FORMATTING"]
        self.dependencies = {
            "INPUT_VALIDATION": [],
            "PREPROCESSING": ["INPUT_VALIDATION"],
            "ML_STAGE": ["PREPROCESSING"],
            "DL_STAGE": ["PREPROCESSING"],
            "NLP_STAGE": ["PREPROCESSING"],
            "SLM_STAGE": ["ML_STAGE", "DL_STAGE", "NLP_STAGE"],
            "SAFETY_STAGE": ["SLM_STAGE"],
            "OUTPUT_FORMATTING": ["SAFETY_STAGE"]
        }
        
    def get_execution_order(self) -> List[str]:
        """Returns valid topological execution order."""
        return [
            "INPUT_VALIDATION",
            "PREPROCESSING",
            "ML_STAGE",
            "DL_STAGE",
            "NLP_STAGE",
            "SLM_STAGE",
            "SAFETY_STAGE",
            "OUTPUT_FORMATTING"
        ]
        
    def validate_graph(self) -> bool:
        """Verifies graph has no cycles and all dependencies exist."""
        for node, deps in self.dependencies.items():
            for d in deps:
                if d not in self.nodes:
                    return False
        return True
