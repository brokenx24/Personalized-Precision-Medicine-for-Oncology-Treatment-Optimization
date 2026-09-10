"""
STAGE 04 — SLM ENGINEER
DATA LAYER: DOMAIN BALANCER & CURRICULUM SAMPLER
Prevents dominant oncology categories from drowning out rare mutations or toxicities.
"""
import random
from collections import defaultdict

class DomainBalancedSampler:
    def __init__(self, records, random_seed=42):
        self.records = records
        self.rng = random.Random(random_seed)
        self.strata = defaultdict(list)
        self._stratify()

    def _stratify(self):
        for idx, rec in enumerate(self.records):
            inp = rec.get("input", "").lower()
            meta = rec.get("metadata", {})
            cancer_type = meta.get("cancer_type", "").lower()
            if not cancer_type:
                for c in ["lung", "breast", "colorectal", "melanoma", "prostate"]:
                    if c in inp:
                        cancer_type = c
                        break
            if not cancer_type:
                cancer_type = "other"
            
            # Identify mutation presence
            has_mut = any(m in inp for m in ["egfr", "kras", "braf", "brca"])
            key = (cancer_type, has_mut)
            self.strata[key].append(idx)

    def sample_batch(self, batch_size=2):
        keys = list(self.strata.keys())
        selected_indices = []
        for _ in range(batch_size):
            chosen_key = self.rng.choice(keys)
            idx = self.rng.choice(self.strata[chosen_key])
            selected_indices.append(idx)
        return [self.records[i] for i in selected_indices]

    def get_strata_distribution(self):
        return {f"{k[0]}_mut_{k[1]}": len(v) for k, v in self.strata.items()}

if __name__ == "__main__":
    from dataset_loader import load_split
    recs = load_split("train")
    sampler = DomainBalancedSampler(recs)
    dist = sampler.get_strata_distribution()
    print("Domain strata distribution:")
    for k, v in dist.items():
        print(f"  {k}: {v}")
