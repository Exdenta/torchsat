
class TrainingMetrics:
    def __init__(self, loss: float = 0.0):
        self.loss = loss

class ValidationMetrics:
    def __init__(self, loss: float = 0.0, precision: float = 0.0, recall: float = 0.0, f1: float = 0.0):
        self.loss = loss
        self.precision = precision
        self.recall = recall
        self.f1 = f1
