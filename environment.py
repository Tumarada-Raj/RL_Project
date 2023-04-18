from common import Action
import copy
import gym
from gym import spaces
import numpy as np
from random import choice
from copy import deepcopy
from utils import stateSpaceVector


class KarelEnv(gym.Env):
    N_ACTIONS = 6
    # Direction encoding

    d_order = ["north", "east", "south", "west"]
    d_mapping = {"north": (-1, 0), "east": (0, 1),
                  "south": (1, 0), "west": (0, -1)}

    def __init__(self, task_space=None):  #, is_compact=True, reward_func='binary'):
        super(KarelEnv, self).__init__()

        self.rewards = self.rewardDesign
        self.task_space = task_space
        #self.is_compact = is_compact
        #self.debug = False
        self.probe_mode = False

        # Each state in a task is specified using 8 (4 pre-grid & 4 post-grid directions) + 2 (Pre-grid and post grid markers) + 1(representing wall) .
        self.obs_shape = (4, 4, 11)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.uint8
        )

        self.actions = {
            Action.move: self.move,
            Action.turnLeft: lambda src_state: self.turn(-1, src_state),
            Action.turnRight: lambda src_state: self.turn(1, src_state),
            Action.pickMarker: self.pickMarker,
            Action.putMarker: self.putMarker,
            Action.finish: self.finish,
        }

    def reset(self, init_state=None):
        if init_state is None:
            init_state = choice(self.task_space)

        self.init(init_state)
        return stateSpaceVector(init_state)

    def init(self, task):
        self.task = copy.deepcopy(task)
        self.is_terminal = False

        self.task["pregrid_markers"] = set(
            map(tuple, self.task["pregrid_markers"]))
        self.task["postgrid_markers"] = set(
            map(tuple, self.task["postgrid_markers"]))

        # Active (i.e., changing) state, note that this is not the total state
        self.state = {
            "r": self.task["pregrid_agent_row"],
            "c": self.task["pregrid_agent_col"],
            "dir": self.task["pregrid_agent_dir"],
            "markers": self.task["pregrid_markers"],
        }

        # Active target state, note that this is not the total state
        self.target_state = {
            "r": self.task["postgrid_agent_row"],
            "c": self.task["postgrid_agent_col"],
            "dir": self.task["postgrid_agent_dir"],
            "markers": self.task["postgrid_markers"],
        }

    def get_full_state(self, state=None):
        if state is None:
            state = self.state

        if state == "terminal":
            return "terminal"

        task_state = copy.deepcopy(self.task)
        task_state["pregrid_agent_row"] = state["r"]
        task_state["pregrid_agent_col"] = state["c"]
        task_state["pregrid_agent_dir"] = state["dir"]
        task_state["pregrid_markers"] = state["markers"]
        return task_state

    def probe(self, action):
        state_copy = deepcopy(self.state)
        self.next_state, self.is_terminal = self.actions[action](state_copy)
        r = self.rewards(self.state, action)
        is_solved = self.state == self.target_state and action == Action.finish
        has_crashed = self.is_terminal and not is_solved

        next_obs = stateSpaceVector(self.get_full_state())
        return next_obs, r, self.is_terminal, {"solved": is_solved, "crashed": has_crashed}

    def step(self, action):
        state_copy = deepcopy(self.state)
        self.next_state, self.is_terminal = self.actions[action](state_copy)
        reward = self.rewards(self.state, action)
        is_solved = self.state == self.target_state and action == Action.finish
        has_crashed = self.is_terminal and not is_solved
        self.state = self.next_state

        next_obs = stateSpaceVector(self.get_full_state())
        return next_obs, reward, self.is_terminal, {"solved": is_solved, "crashed": has_crashed}

    def rewardDesign(self, s, a):

        if s == self.target_state and a == Action.finish:   # Task solved
            return 10

        elif a == Action.putMarker:
            loc = (s["r"], s["c"])
            if loc in self.task["postgrid_markers"] and loc not in s["markers"]:
                return 0
            else:
                return -3
        else:
            return 0

    def move(self, src_state):
        r, c = src_state["r"], src_state["c"]
        dir = src_state["dir"]

        coord = self.d_mapping[dir]
        next_pos = [r + coord[0], c + coord[1]]

        out_of_bounds = (
            next_pos[0] >= self.task["gridsz_num_rows"]
            or next_pos[1] >= self.task["gridsz_num_cols"]
            or next_pos[0] < 0
            or next_pos[1] < 0
        )
        wall_hit = next_pos in self.task["walls"]

        if out_of_bounds or wall_hit:
            return src_state, True

        src_state["r"], src_state["c"] = next_pos[0], next_pos[1]

        return src_state, False

    def turn(self, pos, src_state):
        r, c = src_state["r"], src_state["c"]
        dir = src_state["dir"]

        dir_loc = self.d_order.index(dir)
        next_loc = (dir_loc + pos + 4) % 4
        new_dir = self.d_order[next_loc]

        src_state["dir"] = new_dir
        return src_state, False

    def pickMarker(self, src_state):
        r, c = src_state["r"], src_state["c"]
        if (r, c) not in src_state["markers"]:
            return src_state, True

        src_state["markers"].remove((r, c))
        return src_state, False


    def putMarker(self, src_state):
        r, c = src_state["r"], src_state["c"]
        if (r, c) in src_state["markers"]:
            return src_state, True
        
        src_state["markers"].add((r, c))
        return src_state, False

    def finish(self, src_state):
        return src_state, True
