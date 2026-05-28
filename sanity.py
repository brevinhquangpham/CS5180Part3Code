"""
sanity.py — runs one episode with a random policy and prints each step.
"""

from env import ApartmentEnv
import numpy as np

env = ApartmentEnv(T=4, K=4, seed=42)
obs, info = env.reset()

print(f"{'t':>4} {'U_t':>6} {'action':>8} {'reward':>8} {'done':>6}")
print("-" * 40)

done = False
while not done:
    t = obs["t"]
    u = obs["u"]
    action = env.action_space.sample()  # random policy
    action_str = "accept" if action == 1 else "reject"

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"{t:>4} {u:>6} {action_str:>8} {reward:>8.1f} {done!s:>6}")

print("\nEpisode finished.")
