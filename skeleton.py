# gymnasium>=0.29, torch, numpy, networkx, lightgbm (for alert proxy)
import gymnasium as gym
import numpy as np
import networkx as nx

class IRSEnv(gym.Env):
    def __init__(self, G, traffic_gen, attacker, detector, max_steps=900):
        self.G = G  # networkx graph with node roles
        self.traffic = traffic_gen
        self.attacker = attacker
        self.detector = detector
        self.max_steps = max_steps
        self.step_t = 0

        self.action_space = gym.spaces.Dict({
            "atype": gym.spaces.Discrete(5),      # isolate, throttle, patch, elevate_logging, unquarantine
            "target": gym.spaces.Discrete(G.number_of_nodes()+G.number_of_edges())
        })
        # Flattened obs for starter; replace with a GNN encoder later
        feat_dim = 8  # example
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(G.number_of_nodes()*feat_dim,), dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_t = 0
        self.attacker.reset(self.G)
        self.traffic.reset()
        self.state = self._build_obs()
        return self.state, {}

    def step(self, action):
        # apply action, e.g., isolate node -> remove incident edges temporarily, record costs
        action_cost, downtime_cost = self._apply(action)

        # advance environment
        normal_bytes = self.traffic.tick(self.G)
        exfil_bytes = self.attacker.tick(self.G)
        alerts = self.detector.alerts(self.G)

        # build reward
        reward = -1e-6*exfil_bytes - 0.1*downtime_cost - 0.01*action_cost
        if self.attacker.contained:
            reward += 5.0

        self.step_t += 1
        terminated = self.attacker.finished or self.step_t >= self.max_steps
        truncated = False
        self.state = self._build_obs(alerts=alerts, normal=normal_bytes, exfil=exfil_bytes)
        info = {"exfil_bytes": exfil_bytes}
        return self.state, reward, terminated, truncated, info

    def _apply(self, action):
        # TODO: action logic with cooldowns & parameter masks
        return 1.0, 0.0

    def _build_obs(self, **signals):
        # TODO: concatenate per-node features into a vector
        return np.zeros(self.observation_space.shape, dtype=np.float32)
