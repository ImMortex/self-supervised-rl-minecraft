class EarlyStopping:
    def __init__(self, tolerance=5, target_score=0):

        self.tolerance = tolerance
        self.target_score = target_score
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.tolerance < 0:
            return
        if score >= self.target_score:
            self.counter += 1
            print("early stopping tolerance: " + str(self.tolerance) + " | counter at: " + str(self.counter))
            if self.counter >= self.tolerance:
                self.early_stop = True
