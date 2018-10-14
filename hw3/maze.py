class Maze:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        self.nS = self.WORLD_WIDTH * self.WORLD_HEIGHT
        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.nA = len(self.actions)

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = []
        self.new_obstacles = []

    def encode(self,state):
        return state[0] * self.WORLD_WIDTH + state[1]

    def decode(self,state_en):
        x = state_en // self.WORLD_WIDTH
        y = state_en % self.WORLD_WIDTH
        return [x,y]

    def reset(self):
        self.state = [0,0]
        self.state[:] = self.START_STATE[:]
        return self.encode(self.state)

    def step(self, action):
        x, y = self.state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = self.state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        self.state = [x,y]

        return self.encode([x,y]), reward, done, None   # set up a blocking maze instance
