import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# MFCC feature files to test
feature_sizes = [13, 20, 40, 60]
svm_accuracies = []
valid_feature_sizes = []
best_accuracy = 0
best_feature_size = None
best_model = None
best_y_test = None
best_prediction = None

for num_features in feature_sizes:
    print(f"\nTraining SVM model using {num_features} MFCC features...")

    # Load data
    file_name = f"music_features_{num_features}.npz"

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    valid_feature_sizes.append(num_features)

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

    # feature scaling
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    # Train model
    svm_model.fit(x_train, y_train)

    # Predict test data
    svm_prediction = svm_model.predict(x_test)

    # Calculate accuracy
    svm_accuracy = accuracy_score(y_test, svm_prediction)
    svm_accuracies.append(svm_accuracy)

    print(f"SVM accuracy with {num_features} MFCC features: {svm_accuracy * 100:.2f}%")

    # Save best model result for confusion matrix
    if svm_accuracy > best_accuracy:
        best_accuracy = svm_accuracy
        best_feature_size = num_features
        best_model = svm_model
        best_y_test = y_test
        best_prediction = svm_prediction

if svm_accuracies:
    # Plot SVM accuracy for different MFCC feature sizes
    plt.figure(figsize=(9, 5))
    plt.plot(valid_feature_sizes, svm_accuracies, marker="o", linestyle="-")
    plt.title("SVM accuracy using different numbers of MFCC features", fontsize=12, fontweight="bold")
    plt.xlabel("Number of MFCC features")
    plt.ylabel("Accuracy")
    plt.xticks(valid_feature_sizes)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)

    for i, acc in enumerate(svm_accuracies):
        plt.text(valid_feature_sizes[i], acc + 0.01, f"{acc * 100:.1f}%", ha="center")

    plt.savefig("svm_accuracy_by_features.png", dpi=300)
    plt.show()

    print("\nBest SVM result:")
    print(f"Best feature size: {best_feature_size}")
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    print("\nClassification report for best SVM model:")
    print(classification_report(best_y_test, best_prediction))

    # Confusion matrix for best SVM model
    labels = sorted(np.unique(best_y_test))
    cm = confusion_matrix(best_y_test, best_prediction, labels=labels)
    
    # Explicitly pass the subplot axis to ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    
    plt.title(f"SVM Confusion Matrix ({best_feature_size} MFCC features)")
    plt.tight_layout()
    plt.savefig(f"svm_confusion_matrix_{best_feature_size}.png", dpi=300)
    plt.show()
else:
    print("No valid feature data files found. Exiting program without plots.")
