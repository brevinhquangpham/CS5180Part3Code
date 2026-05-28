import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ApartmentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, T: int, K: int, seed=None, noise_std: float = 0.0):
        super().__init__()
        self.T = T
        self.K = K
        self.noise_std = noise_std

        # action space: 0 = reject, 1 = accept
        self.action_space = spaces.Discrete(2)

        # Observation space: (t, U_t) where t in [1,T], U_t in [1,K]
        # In noisy mode the observed quality is continuous, so we use Box.
        if noise_std == 0.0:
            self.observation_space = spaces.Dict(
                {
                    "t": spaces.Discrete(T + 1, start=1),  # weeks 1-T
                    "u": spaces.Discrete(K + 1, start=1),  # qualities 1-K
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "t": spaces.Discrete(T + 1, start=1),
                    "u": spaces.Box(
                        low=1.0 - 10 * noise_std,
                        high=K + 10 * noise_std,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )

        self._rng = np.random.default_rng(seed)
        self._t = None
        self._true_u = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 1
        self._true_u = int(self._rng.integers(1, self.K + 1))
        obs = self._make_obs()
        return obs, {}

    def step(self, action: int):
        assert self._t is not None, "Call reset() before step()."
        assert action in (0, 1), f"Invalid action {action}."

        terminated = False
        truncated = False
        reward = 0.0

        if action == 1:
            # accept
            reward = float(self._true_u)
            terminated = True
            obs = self._make_obs()  # final observ (won't be used by agent)
            info = {"true_u": self._true_u, "accepted_week": self._t}
            return obs, reward, terminated, truncated, info

        # Reject
        if self._t == self.T:
            # FEll back to sublet
            reward = 0.0
            terminated = True
            obs = self._make_obs()
            info = {"true_u": self._true_u, "sublet": True}
            return obs, reward, terminated, truncated, info

        # move To next week
        self._t += 1
        self._true_u = int(self._rng.integers(1, self.K + 1))
        obs = self._make_obs()
        info = {"true_u": self._true_u}
        return obs, reward, terminated, truncated, info

    def _make_obs(self):
        if self.noise_std == 0.0:
            return {"t": self._t, "u": self._true_u}
        else:
            noisy_u = float(self._true_u) + float(self._rng.normal(0, self.noise_std))
            return {"t": self._t, "u": np.array([noisy_u], dtype=np.float32)}
