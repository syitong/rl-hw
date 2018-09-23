class RandomWalk:
    def __init__(self):
        self.initial = 3
        self.reward = 1

    def reset(self):
        self.current = self.initial
        self.done = False
        return self.current

    def step(self,action):
        if self.current == 0 or self.current == 6:
            self.done = True
            return self.current, 0, self.done, None
        if action == 0:
            self.current -= 1
        else:
            self.current += 1
        if self.current == 0:
            self.done = True
            return self.current, 0, self.done, None
        elif self.current == 6:
            self.done = True
            return self.current, self.reward, self.done, None
        else:
            return self.current, 0, self.done, None
