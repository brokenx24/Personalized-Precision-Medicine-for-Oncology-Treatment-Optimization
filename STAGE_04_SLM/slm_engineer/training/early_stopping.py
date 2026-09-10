"""
STAGE 04 — SLM ENGINEER
TRAINING LAYER: EARLY STOPPING & PATIENCE MONITOR
Prevents overfitting by tracking validation composite loss and stopping when performance plateaus.
"""
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.005, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 1

    def step(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True, "Initial best checkpoint"

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True, f"Score improved to {score:.4f} at epoch {epoch}"
        else:
            self.counter += 1
            msg = f"No improvement for {self.counter}/{self.patience} epochs"
            if self.counter >= self.patience:
                self.early_stop = True
                msg += " -> Triggering Early Stopping"
            return False, msg
