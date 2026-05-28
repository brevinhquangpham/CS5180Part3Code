import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def run_episodes(policy, T: int, K: int, N: int, noise_std: float = 0.0, seed: int = 0):
    env = ApartmentEnv(T=T, K=K, seed=seed, noise_std=noise_std)
    returns = np.empty(N)
    for i in range(N):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns[i] = total
    return returns


def report(name: str, returns: np.ndarray, T: int):
    mean = returns.mean()
    se = returns.std() / np.sqrt(len(returns))
    frac_zero = (returns == 0).mean()
    print(f"  {name:<30s}  mean={mean:.4f} ± {se:.4f}   sublet_frac={frac_zero:.4f}")
    return mean, se, frac_zero


# ──────────────────────────────────────────────────────────────────────────────
# Part (c)
# ──────────────────────────────────────────────────────────────────────────────

T, K, N = 4, 4, 10_000
print("=" * 65)
print(f"Part (c) — noiseless (σ=0), T={T}, K={K}, N={N}")
print("=" * 65)

policies_c = {
    "Random (p=1/T)": RandomPolicy(T),
    "Threshold(u_min=1)": ThresholdPolicy(1),
    "Threshold(u_min=2)": ThresholdPolicy(2),
    "Threshold(u_min=3)": ThresholdPolicy(3),
    "Threshold(u_min=4)": ThresholdPolicy(4),
    "Optimal": OptimalPolicy(),
}

all_returns = {}
for name, policy in policies_c.items():
    returns = run_episodes(policy, T, K, N, noise_std=0.0, seed=7)
    all_returns[name] = returns
    report(name, returns, T)

# Find best threshold
threshold_means = {
    u: all_returns[f"Threshold(u_min={u})"].mean() for u in range(1, K + 1)
}
best_u = max(threshold_means, key=threshold_means.get)
print(f"\n  Best fixed threshold: u_min={best_u}  (mean={threshold_means[best_u]:.4f})")
optimal_mean = all_returns["Optimal"].mean()
print(f"  Optimal policy mean : {optimal_mean:.4f}")
print(f"  Gap                 : {optimal_mean - threshold_means[best_u]:.4f}")

# Histogram figure — part (c)
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
plot_keys = ["Random (p=1/T)", f"Threshold(u_min={best_u})", "Optimal"]
colors = ["steelblue", "darkorange", "seagreen"]

for ax, key, color in zip(axes, plot_keys, colors):
    data = all_returns[key]
    ax.hist(
        data,
        bins=np.arange(0, K + 2) - 0.5,
        color=color,
        edgecolor="white",
        alpha=0.85,
        density=True,
    )
    ax.set_title(key, fontsize=10)
    ax.set_xlabel("Return (apartment quality)")
    ax.set_xticks(range(0, K + 1))

axes[0].set_ylabel("Density")
fig.suptitle(f"Return distributions — noiseless (T={T}, K={K}, N={N})", fontsize=12)
plt.tight_layout()
plt.savefig("/home/brevinh/School/RL/PSet0/Code/yea.png", dpi=150)
plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Part (d) — robustness to noise
# ──────────────────────────────────────────────────────────────────────────────

sigmas = [0.0, 0.5, 1.0, 2.0]
print("\n" + "=" * 65)
print(f"Part (d) — noise robustness sweep σ ∈ {sigmas}")
print("=" * 65)

policies_d = {
    "Random": RandomPolicy(T),
    f"Threshold(u_min={best_u})": ThresholdPolicy(best_u),
    "Optimal": OptimalPolicy(),
}

noise_results = {name: [] for name in policies_d}

for sigma in sigmas:
    print(f"\n  sig = {sigma}")
    for name, policy in policies_d.items():
        returns = run_episodes(policy, T, K, N, noise_std=sigma, seed=13)
        mean, se, _ = report(name, returns, T)
        noise_results[name].append(mean)

# Line plot — part (d)
fig, ax = plt.subplots(figsize=(7, 4))
markers = ["o", "s", "^"]
colors_d = ["steelblue", "darkorange", "seagreen"]

for (name, means), marker, color in zip(noise_results.items(), markers, colors_d):
    ax.plot(sigmas, means, marker=marker, label=name, color=color, linewidth=2)

ax.set_xlabel("Noise std σ", fontsize=11)
ax.set_ylabel("Mean return", fontsize=11)
ax.set_title(
    f"Policy robustness to observation noise (T={T}, K={K}, N={N})", fontsize=11
)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/brevinh/School/RL/PSet0/Code/noise_robustness.png", dpi=150)
plt.close()
print("\n  Saved: returns_noise_robustness.png")
print("\nDone.")
