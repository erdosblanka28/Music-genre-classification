import numpy as np
import matplotlib.pyplot as plt

# FINAL COMPARISON PLOTS FROM ALREADY MEASURED RESULTS

feature_sizes = [13, 20, 40, 60]

results = {
    "KNN": [0.555, 0.560, 0.570, 0.580],
    "SVM": [0.640, 0.650, 0.670, 0.655],
    "Random Forest": [0.630, 0.645, 0.660, 0.635],
    "Neural Network": [0.630, 0.605, 0.640, 0.585]
}
model_names = list(results.keys())

# Plot 1: Accuracy vs MFCC features
plt.figure(figsize=(10, 6))
for model_name in model_names:
    plt.plot(
        feature_sizes,
        results[model_name],
        marker="o",
        linestyle="-",
        label=model_name
    )

plt.title("Model accuracy comparison using different numbers of MFCC features", fontsize=12, fontweight="bold")
plt.xlabel("Number of MFCC features")
plt.ylabel("Accuracy")
plt.xticks(feature_sizes)
plt.ylim(0, 1)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

# offset to fix overlap
text_offsets = {
    "KNN": -0.048,
    "SVM": 0.042,
    "Random Forest": 0.012,
    "Neural Network": -0.030
}

line_colors = {"KNN": "C0", "SVM": "C1", "Random Forest": "C2", "Neural Network": "C3"}

for model_name in model_names:
    for i, acc in enumerate(results[model_name]):
        plt.text(
            feature_sizes[i], 
            acc + text_offsets[model_name], 
            f"{acc * 100:.1f}%", 
            ha="center", 
            va="center",
            fontsize=8,
            color=line_colors[model_name],
            fontweight="bold"
        )

plt.savefig("comparison_all_models_by_features.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: Best accuracy by model
best_accuracies = [max(results[model_name]) for model_name in model_names]
best_feature_sizes = [
    feature_sizes[np.argmax(results[model_name])]
    for model_name in model_names
]

plt.figure(figsize=(9, 5))
plt.bar(model_names, best_accuracies)
plt.title("Best accuracy achieved by each model", fontsize=12, fontweight="bold")
plt.xlabel("Model")
plt.ylabel("Best accuracy")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle=":", alpha=0.6)
for i, acc in enumerate(best_accuracies):
    plt.text(
        i,
        acc + 0.01,
        f"{acc * 100:.1f}%\n({best_feature_sizes[i]} MFCC)",
        ha="center"
    )

plt.savefig("comparison_best_accuracy_by_model.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 3: Direct comparison at 40 MFCC features
feature_40_index = feature_sizes.index(40)
accuracies_40 = [results[model_name][feature_40_index] for model_name in model_names]

plt.figure(figsize=(9, 5))
plt.bar(model_names, accuracies_40)
plt.title("Model comparison using 40 MFCC features", fontsize=12, fontweight="bold")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle=":", alpha=0.6)
for i, acc in enumerate(accuracies_40):
    plt.text(i, acc + 0.01, f"{acc * 100:.1f}%", ha="center")

plt.savefig("comparison_40_mfcc_features.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 4: Accuracy heatmap
accuracy_matrix = np.array([
    [results[model_name][i] * 100 for i in range(len(feature_sizes))]
    for model_name in model_names
])

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(accuracy_matrix, aspect="auto")
ax.set_title("Accuracy heatmap: models vs MFCC feature sizes", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of MFCC features")
ax.set_ylabel("Model")
ax.set_xticks(np.arange(len(feature_sizes)))
ax.set_xticklabels(feature_sizes)
ax.set_yticks(np.arange(len(model_names)))
ax.set_yticklabels(model_names)

for i in range(len(model_names)):
    for j in range(len(feature_sizes)):
        ax.text(j, i, f"{accuracy_matrix[i, j]:.1f}%", ha="center", va="center")

fig.colorbar(im, ax=ax, label="Accuracy (%)")
plt.tight_layout()
plt.savefig("comparison_accuracy_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Final comparison plots saved successfully.")
