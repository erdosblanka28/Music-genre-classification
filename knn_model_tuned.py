import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#TUNED KNN MODEL FOR MUSIC GENRE CLASSIFICATION

#This is the fair KNN version for comparison: scaled input + optimized k + optimized distance metric. 
SHOW_PLOTS = True

feature_sizes = [13, 20, 40, 60]
tested_feature_sizes = []
knn_accuracies = []
best_params_list = []
best_accuracy = 0
best_feature_size = None
best_model = None
best_y_test = None
best_prediction = None
best_label_encoder = None
best_params = None

for num_features in feature_sizes:
    print(f"\nTraining tuned KNN model using {num_features} MFCC features...")

    file_name = f"music_features_{num_features}.npz"

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    saved_data = np.load(file_name)
    x = saved_data["features"]
    y = saved_data["labels"]

    # Encode labels into numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Data split: 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Cross-validation setup for parameter tuning
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # StandardScaler because KNN is distance-based, and we want to ensure that all features contribute equally to the distance calculations
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    param_grid = {
        "knn__n_neighbors": list(range(1, 51)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }

    # Grid search finds the best KNN settings using only the training set
    knn_grid = GridSearchCV(
        knn_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    # Train and tune model
    knn_grid.fit(x_train, y_train)

    # Best model prediction on the test set
    knn_prediction = knn_grid.predict(x_test)

    # Calculate test accuracy
    knn_accuracy = accuracy_score(y_test, knn_prediction)

    tested_feature_sizes.append(num_features)
    knn_accuracies.append(knn_accuracy)
    best_params_list.append(knn_grid.best_params_)

    print(f"Best parameters for {num_features} MFCC features:")
    print(knn_grid.best_params_)
    print(f"Tuned KNN accuracy with {num_features} MFCC features: {knn_accuracy * 100:.2f}%")

    # Save best overall KNN result
    if knn_accuracy > best_accuracy:
        best_accuracy = knn_accuracy
        best_feature_size = num_features
        best_model = knn_grid
        best_y_test = y_test
        best_prediction = knn_prediction
        best_label_encoder = label_encoder
        best_params = knn_grid.best_params_

# 1. Plot tuned KNN accuracy for different MFCC feature sizes

plt.figure(figsize=(9, 5))
plt.plot(tested_feature_sizes, knn_accuracies, marker="o", linestyle="-")

plt.title("Tuned KNN accuracy using different numbers of MFCC features", fontsize=12, fontweight="bold")
plt.xlabel("Number of MFCC features")
plt.ylabel("Accuracy")
plt.xticks(tested_feature_sizes)
plt.ylim(0, 1)
plt.grid(True, linestyle=":", alpha=0.6)

for i, acc in enumerate(knn_accuracies):
    plt.text(tested_feature_sizes[i], acc + 0.01, f"{acc * 100:.1f}%", ha="center")

plt.savefig("knn_tuned_accuracy_by_features.png", dpi=300, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# Save KNN results to CSV

with open("knn_tuned_results.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "MFCC features",
        "Accuracy",
        "Best k",
        "Best weights",
        "Best metric"
    ])

    for i, num_features in enumerate(tested_feature_sizes):
        params = best_params_list[i]

        writer.writerow([
            num_features,
            knn_accuracies[i],
            params["knn__n_neighbors"],
            params["knn__weights"],
            params["knn__metric"]
        ])


# Print best KNN result

print("\nBest tuned KNN result:")
print(f"Best feature size: {best_feature_size}")
print(f"Best accuracy: {best_accuracy * 100:.2f}%")
print("Best parameters:")
print(best_params)


# Classification report for best KNN model

class_names = best_label_encoder.classes_

print("\nClassification report for best tuned KNN model:")
print(classification_report(
    best_y_test,
    best_prediction,
    target_names=class_names
))


# 2. Confusion matrix for best KNN model

labels = np.arange(len(class_names))

cm = confusion_matrix(best_y_test, best_prediction, labels=labels)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)

plt.title(f"Tuned KNN Confusion Matrix ({best_feature_size} MFCC features)")
plt.tight_layout()
plt.savefig(f"knn_tuned_confusion_matrix_{best_feature_size}.png", dpi=300, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# 3. Best k value for each MFCC feature size

best_k_values = [
    params["knn__n_neighbors"]
    for params in best_params_list
]

plt.figure(figsize=(9, 5))
plt.plot(tested_feature_sizes, best_k_values, marker="o", linestyle="-")

plt.title("Best k value for tuned KNN", fontsize=12, fontweight="bold")
plt.xlabel("Number of MFCC features")
plt.ylabel("Best k value")
plt.xticks(tested_feature_sizes)
plt.grid(True, linestyle=":", alpha=0.6)

for i, k in enumerate(best_k_values):
    plt.text(tested_feature_sizes[i], k + 0.5, str(k), ha="center")

plt.savefig("knn_tuned_best_k_by_features.png", dpi=300, bbox_inches="tight")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close()