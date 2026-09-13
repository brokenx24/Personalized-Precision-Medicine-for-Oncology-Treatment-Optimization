"""
Genomic Mutation Profile Generator.
Synthesizes Level 1 to Level 5 genomic mutations categorized by biological evidence status.
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class MutationGenerator:
    def __init__(self, stats_path: Optional[Path] = None, cooc_path: Optional[Path] = None):
        self.stats_path = stats_path or (
            HOSPITAL_ROOT / "STAGE_05_GENAI" / "data" / "mutation_statistics" / "mutation_statistics.json"
        )
        self.cooc_path = cooc_path or (
            HOSPITAL_ROOT / "STAGE_05_GENAI" / "data" / "mutation_statistics" / "mutation_cooccurrence.json"
        )
        self.stats = self._load_json(self.stats_path)
        self.cooc = self._load_json(self.cooc_path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def generate_mutations(self, patient_id: str, cancer_type: str, difficulty_level: str = "LEVEL_1") -> Dict[str, Any]:
        pool = self.stats.get("cancer_specific_mutations", {}).get(cancer_type, [])
        if not pool:
            # Fallback pool if cancer type not explicitly indexed
            pool = [
                {"gene": "TP53", "variant": "R273H", "level": "LEVEL_1", "evidence": "CLINICAL_TRIAL_EVIDENCE"},
                {"gene": "KRAS", "variant": "G12D", "level": "LEVEL_1", "evidence": "FDA_APPROVED_BIOMARKER"},
                {"gene": "PIK3CA", "variant": "E545K", "level": "LEVEL_2", "evidence": "CLINICAL_TRIAL_EVIDENCE"}
            ]

        mutations_list = []
        if difficulty_level == "LEVEL_1":
            # Single canonical driver
            candidates = [m for m in pool if m.get("level") == "LEVEL_1"] or pool[:1]
            chosen = random.choice(candidates)
            mutations_list.append({
                "mutation_id": f"MUT-{chosen['gene']}-01",
                "gene": chosen["gene"],
                "variant": chosen["variant"],
                "variant_type": "SNV" if "del" not in chosen["variant"].lower() else "Deletion",
                "allele_frequency_pct": round(random.uniform(15.0, 48.0), 1),
                "classification": "KNOWN_REFERENCE",
                "evidence_status": chosen.get("evidence", "FDA_APPROVED_BIOMARKER"),
                "scenario_role": "PRIMARY_DRIVER",
                "confidence": 0.98,
                "is_hypothetical": False,
                "associated_therapies": ["Targeted Standard of Care"]
            })

        elif difficulty_level == "LEVEL_2":
            # Uncommon driver
            candidates = [m for m in pool if m.get("level") in ["LEVEL_1", "LEVEL_2"]] or pool
            chosen = random.choice(candidates)
            mutations_list.append({
                "mutation_id": f"MUT-{chosen['gene']}-02",
                "gene": chosen["gene"],
                "variant": chosen["variant"],
                "variant_type": "Fusion" if "fusion" in chosen["variant"].lower() else "SNV",
                "allele_frequency_pct": round(random.uniform(10.0, 35.0), 1),
                "classification": "KNOWN_REFERENCE",
                "evidence_status": chosen.get("evidence", "FDA_APPROVED_BIOMARKER"),
                "scenario_role": "PRIMARY_DRIVER",
                "confidence": 0.92,
                "is_hypothetical": False,
                "associated_therapies": ["Next-gen Targeted Inhibitor"]
            })

        elif difficulty_level == "LEVEL_3":
            # Rare alteration
            chosen = pool[-2] if len(pool) >= 2 else pool[0]
            mutations_list.append({
                "mutation_id": f"MUT-{chosen['gene']}-03",
                "gene": chosen["gene"],
                "variant": f"{chosen['variant']} + Secondary Polymorphism",
                "variant_type": "Complex",
                "allele_frequency_pct": round(random.uniform(4.5, 18.0), 1),
                "classification": "SYNTHETIC_VARIATION",
                "evidence_status": "CLINICAL_TRIAL_EVIDENCE",
                "scenario_role": "BORDERLINE_MODIFIER",
                "confidence": 0.82,
                "is_hypothetical": False,
                "associated_therapies": ["Clinical Trial Regimen"]
            })

        elif difficulty_level == "LEVEL_4":
            # Compound co-mutation (e.g. Primary Driver + Co-mutation challenge)
            mutations_list.append({
                "mutation_id": "MUT-COMP-01",
                "gene": "EGFR" if "Lung" in cancer_type else "KRAS",
                "variant": "Exon 19 Deletion" if "Lung" in cancer_type else "G12D",
                "variant_type": "Deletion" if "Lung" in cancer_type else "SNV",
                "allele_frequency_pct": 34.5,
                "classification": "KNOWN_REFERENCE",
                "evidence_status": "FDA_APPROVED_BIOMARKER",
                "scenario_role": "PRIMARY_DRIVER",
                "confidence": 0.95,
                "is_hypothetical": False
            })
            mutations_list.append({
                "mutation_id": "MUT-COMP-02",
                "gene": "PIK3CA",
                "variant": "H1047R",
                "variant_type": "SNV",
                "allele_frequency_pct": 21.0,
                "classification": "KNOWN_REFERENCE",
                "evidence_status": "CONTRADICTORY_EVIDENCE",
                "scenario_role": "CO_MUTATION_CHALLENGE",
                "confidence": 0.88,
                "is_hypothetical": False,
                "resistance_mechanism": "PI3K/AKT downstream pathway activation bypassing upstream inhibition"
            })

        else:  # LEVEL_5
            # Tertiary acquired resistance compound scenario
            mutations_list.append({
                "mutation_id": "MUT-RES-01",
                "gene": "EGFR",
                "variant": "Exon 19 Deletion (E746_A750del)",
                "variant_type": "Deletion",
                "allele_frequency_pct": 38.0,
                "classification": "KNOWN_REFERENCE",
                "evidence_status": "FDA_APPROVED_BIOMARKER",
                "scenario_role": "PRIMARY_DRIVER",
                "confidence": 0.99,
                "is_hypothetical": False
            })
            mutations_list.append({
                "mutation_id": "MUT-RES-02",
                "gene": "EGFR",
                "variant": "T790M",
                "variant_type": "SNV",
                "allele_frequency_pct": 22.5,
                "classification": "KNOWN_REFERENCE",
                "evidence_status": "FDA_APPROVED_BIOMARKER",
                "scenario_role": "ACQUIRED_RESISTANCE",
                "confidence": 0.96,
                "is_hypothetical": False,
                "resistance_mechanism": "Gatekeeper mutation causing steric hindrance to 1st/2nd gen TKIs"
            })
            mutations_list.append({
                "mutation_id": "MUT-RES-03",
                "gene": "EGFR",
                "variant": "C797S (cis-allelic)",
                "variant_type": "SNV",
                "allele_frequency_pct": 14.2,
                "classification": "KNOWN_REFERENCE",
                "evidence_status": "CLINICAL_TRIAL_EVIDENCE",
                "scenario_role": "ACQUIRED_RESISTANCE",
                "confidence": 0.90,
                "is_hypothetical": False,
                "resistance_mechanism": "Ablates covalent binding of osimertinib; cis-conformation prevents dual TKI response"
            })

        return {
            "profile_id": f"PROF-MUT-{patient_id}",
            "patient_id": patient_id,
            "difficulty_level": difficulty_level,
            "mutations": mutations_list
        }

if __name__ == "__main__":
    mg = MutationGenerator()
    muts = mg.generate_mutations("SYN-PAT-00001", "Lung Adenocarcinoma", "LEVEL_4")
    print("Sample Level 4 Compound mutations:", json.dumps(muts, indent=2))
