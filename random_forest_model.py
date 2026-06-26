import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

SHOW_PLOTS = False

# MFCC feature files to test
feature_sizes = [13, 20, 40, 60]
tested_feature_sizes = []
rf_accuracies = []
best_accuracy = 0
best_feature_size = None
best_model = None
best_y_test = None
best_prediction = None

for num_features in feature_sizes:
    print(f"\nTraining Random Forest model using {num_features} MFCC features...")

    # Load data
    file_name = f"music_features_{num_features}.npz"

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    saved_data = np.load(file_name)
    x = saved_data["features"]
    y = saved_data["labels"]

    # Data split: 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Random Forest does not require feature scaling.
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    # Train model
    rf_model.fit(x_train, y_train)

    # Predict test data
    rf_prediction = rf_model.predict(x_test)

    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_prediction)

    tested_feature_sizes.append(num_features)
    rf_accuracies.append(rf_accuracy)

    print(f"Random Forest accuracy with {num_features} MFCC features: {rf_accuracy * 100:.2f}%")

    # Save best model result for confusion matrix later
    if rf_accuracy > best_accuracy:
        best_accuracy = rf_accuracy
        best_feature_size = num_features
        best_model = rf_model
        best_y_test = y_test
        best_prediction = rf_prediction

if rf_accuracies: 
    # 1. Plot Random Forest accuracy for different MFCC feature sizes 
    plt.figure(figsize=(9, 5))
    plt.plot(tested_feature_sizes, rf_accuracies, marker="o", linestyle="-")
    plt.title("Random Forest accuracy using different numbers of MFCC features", fontsize=12, fontweight="bold")
    plt.xlabel("Number of MFCC features")
    plt.ylabel("Accuracy")
    plt.xticks(tested_feature_sizes)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)

    for i, acc in enumerate(rf_accuracies):
        plt.text(tested_feature_sizes[i], acc + 0.01, f"{acc * 100:.1f}%", ha="center")
    plt.savefig("random_forest_accuracy_by_features.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # 2. Text Classification report for best Random Forest model 
    print("\nBest Random Forest result:")
    print(f"Best feature size: {best_feature_size}")
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    print("\nClassification report for best Random Forest model:")
    print(classification_report(best_y_test, best_prediction))

    # 3. Confusion matrix for best Random Forest model 
    labels = sorted(np.unique(best_y_test))
    cm = confusion_matrix(best_y_test, best_prediction, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"Random Forest Confusion Matrix ({best_feature_size} MFCC features)")
    plt.tight_layout()
    plt.savefig(f"random_forest_confusion_matrix_{best_feature_size}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # 4. Plot Feature Importances
    importances = best_model.feature_importances_
    feature_numbers = np.arange(1, best_feature_size + 1)
    plt.figure(figsize=(10, 5))
    plt.bar(feature_numbers, importances, color="skyblue", edgecolor="black", alpha=0.8)
    plt.title(f"Random Forest feature importances ({best_feature_size} MFCC features)", fontsize=12, fontweight="bold")
    plt.xlabel("MFCC feature number")
    plt.ylabel("Importance")
    
    # Dynamically adapt x-ticks step size to avoid overlapping text on 40/60 dimensions
    if best_feature_size <= 20:
        plt.xticks(feature_numbers)
    else:
        plt.xticks(np.arange(0, best_feature_size + 1, 5)) # Label every 5th feature
        
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.savefig(f"random_forest_feature_importance_{best_feature_size}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

else:
    print("No valid feature files were found. Exiting pipeline.")