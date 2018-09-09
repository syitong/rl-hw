import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment testing the effect of discount factor on finding
    shortest path. Your goal is to reach the exits as soon as possible.

    The grid looks as follows:

    o  o  o  o
    o  o  o  o
    o  o  o  T
    o  T  o  o

    Ts are the exits and x is the location of the agent.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of 1 after the step to an exit state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,5]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 14 or s == 17

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, 0., True)]
                P[s][RIGHT] = [(1.0, s, 0., True)]
                P[s][DOWN] = [(1.0, s, 0., True)]
                P[s][LEFT] = [(1.0, s, 0., True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, is_done(ns_up), is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, is_done(ns_right), is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, is_done(ns_down), is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, is_done(ns_left), is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            is_done = lambda s: s == 14 or s == 17

            if self.s == s:
                output = " x "
            elif is_done(s):
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
