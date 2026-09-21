"""Build + cache Colored-MNIST and save a sample figure.

Usage: uv run python scripts/make_data.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib

matplotlib.use("Agg")

from cmnist.data import build_colored_mnist, save_envs
from cmnist.viz import plot_samples

envs = build_colored_mnist(data_dir="data", seed=0)
save_envs(envs, "data/colored")
for name, env in envs.items():
    print(name, "x:", tuple(env["x"].shape), "P(y=1):", round(float(env["y"].mean()), 3),
          "P(color=1):", round(float((env['color'] == 1).float().mean()), 3))
plot_samples(envs["train_e1"], path="outputs/figs/samples_train.png")
plot_samples(envs["test"], path="outputs/figs/samples_test.png")
print("saved data/colored/*.pt + outputs/figs/samples_*.png")
