import numpy as np


class RandomPolicy:
    """
    Accept with probability 1/T each week, reject otherwise.
    Requires T to be passed at construction to fix probability
    """

    def __init__(self, T: int):
        self.p_accept = 1.0 / T
        self._rng = np.random.default_rng()

    def act(self, obs: dict) -> int:
        return int(self._rng.random() < self.p_accept)


class ThresholdPolicy:
    """
    Accept iff observed U_t >= u_min, reject otherwise.
    """

    def __init__(self, u_min: float):
        self.u_min = u_min

    def act(self, obs: dict) -> int:
        u = obs["u"]
        # Handle both discrete (int) and continuous (np.ndarray) observations
        if isinstance(u, np.ndarray):
            u = float(u[0])
        return int(u >= self.u_min)


class OptimalPolicy:
    """
    Hardcoded optimal policy from Problem 1(c)

    Thresholds W_{t+1} computed by backward induction:
        W_4 = 2.5  →  accept at t=4 if u >= 1  (always accept)
        W_3 = 2.5  →  accept at t=3 if u >= 3
        W_2 = 3.0  →  accept at t=2 if u >= 3
        W_1 = 3.25 →  accept at t=1 if u >= 4
    """

    def __init__(self):
        # Build lookup table for T=4, K=4
        # accept_threshold[t] = minimum u to accept at week t
        self._threshold = {
            1: 4,  # accept only u=4
            2: 3,  # accept u=3,4
            3: 3,  # accept u=3,4
            4: 1,  # always accept
        }

    def act(self, obs: dict) -> int:
        t = int(obs["t"])
        u = obs["u"]
        if isinstance(u, np.ndarray):
            u = float(u[0])
        u_int = int(round(float(u)))
        threshold = self._threshold.get(t, 1)
        return int(u_int >= threshold)
