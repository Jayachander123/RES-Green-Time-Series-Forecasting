
class PageHinkley:
    def __init__(self, delta=0.005, lambd=50.0, alpha=0.99):
        self.delta, self.lambd, self.alpha = delta, lambd, alpha
        self.reset()

    def reset(self, *_):         
        self.mean = 0.0
        self.m    = 0.0

    # add **_  ⇓⇓⇓
    def should_retrain(self, benefit=None, cost_sec=None,
                       error_t=None, **_) -> bool:
        x = benefit if error_t is None else error_t
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x
        self.m    = max(0.0, self.m + x - self.mean - self.delta)
        if self.m > self.lambd:
            self.reset()
            return True
        return False
