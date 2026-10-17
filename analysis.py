"""Iris morphology analysis.

Research question: how well do petal measurements separate the three Iris
species, and which measurements are most discriminative?

Dataset: the classic Iris dataset (Fisher, 1936), loaded from scikit-learn.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "cisc7204"
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

COLORS = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}


def load_data() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    frame = iris.frame.rename(columns={"target": "species_id"})
    frame["species"] = frame["species_id"].map(dict(enumerate(iris.target_names)))
    return frame.drop(columns=["species_id"])


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("species")[["sepal length (cm)", "sepal width (cm)",
                                  "petal length (cm)", "petal width (cm)"]]
        .agg(["mean", "std"])
        .round(2)
    )


def plot_petal_scatter(frame: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for species, group in frame.groupby("species"):
        ax.scatter(
            group["petal length (cm)"],
            group["petal width (cm)"],
            s=45,
            alpha=0.8,
            label=species,
            color=COLORS[species],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title("Petal length vs. petal width by Iris species")
    ax.legend(title="Species")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "petal_scatter.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_petal_box(frame: pd.DataFrame) -> Path:
    species = ["setosa", "versicolor", "virginica"]
    data = [frame.loc[frame["species"] == s, "petal length (cm)"] for s in species]
    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(data, tick_labels=species, patch_artist=True, widths=0.55)
    for patch, s in zip(box["boxes"], species):
        patch.set_facecolor(COLORS[s])
        patch.set_alpha(0.75)
    ax.set_ylabel("Petal length (cm)")
    ax.set_title("Distribution of petal length by species")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "petal_box.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def evaluate_separability(frame: pd.DataFrame) -> float:
    features = frame[["petal length (cm)", "petal width (cm)"]]
    labels = frame["species"]
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, features, labels, cv=cv)
    return float(scores.mean())


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    frame = load_data()
    summary = summarize(frame)
    summary.to_csv(RESULTS_DIR / "species_summary.csv")

    correlation = frame["petal length (cm)"].corr(frame["petal width (cm)"])
    accuracy = evaluate_separability(frame)

    scatter = plot_petal_scatter(frame)
    box = plot_petal_box(frame)

    print("Iris dataset summary (mean / std per species):")
    print(summary.to_string())
    print(f"\nPetal length vs. petal width correlation: r = {correlation:.3f}")
    print(f"5-fold CV accuracy (decision tree on petal features): {accuracy:.3f}")
    print(f"Saved figures: {scatter.name}, {box.name}")
    print(f"Saved summary table: {RESULTS_DIR / 'species_summary.csv'}")


if __name__ == "__main__":
    main()
