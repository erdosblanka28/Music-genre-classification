import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# This is an MLP neural network using the already extracted MFCC features
SHOW_PLOTS =True

feature_sizes = [13, 20, 40, 60]
tested_feature_sizes = []
nn_accuracies = []
best_accuracy = 0
best_feature_size = None
best_model = None
best_y_test = None
best_prediction = None
best_label_encoder = None

for num_features in feature_sizes:
    print(f"\nTraining Neural Network model using {num_features} MFCC features...")

    file_name = f"music_features_{num_features}.npz"

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    saved_data = np.load(file_name)
    x = saved_data["features"]
    y = saved_data["labels"]

    # Convert text labels into numeric labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Scaling
    nn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42
        ))
    ])

    nn_model.fit(x_train, y_train)
    nn_prediction = nn_model.predict(x_test)
    nn_accuracy = accuracy_score(y_test, nn_prediction)
    tested_feature_sizes.append(num_features)
    nn_accuracies.append(nn_accuracy)

    print(f"Neural Network accuracy with {num_features} MFCC features: {nn_accuracy * 100:.2f}%")

    if nn_accuracy > best_accuracy:
        best_accuracy = nn_accuracy
        best_feature_size = num_features
        best_model = nn_model
        best_y_test = y_test
        best_prediction = nn_prediction
        best_label_encoder = label_encoder

if nn_accuracies:

    # 1. Plot Neural Network accuracy for different MFCC feature sizes
    plt.figure(figsize=(9, 5))
    plt.plot(tested_feature_sizes, nn_accuracies, marker="o", linestyle="-", color="C0")
    plt.title("Neural Network accuracy using different numbers of MFCC features", fontsize=12, fontweight="bold")
    plt.xlabel("Number of MFCC features")
    plt.ylabel("Accuracy")
    plt.xticks(tested_feature_sizes)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)

    for i, acc in enumerate(nn_accuracies):
        plt.text(tested_feature_sizes[i], acc + 0.01, f"{acc * 100:.1f}%", ha="center")

    plt.savefig("neural_network_accuracy_by_features.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # 2. Text Classification report for best Neural Network model 
    class_names = best_label_encoder.classes_

    print("\nBest Neural Network result:")
    print(f"Best feature size: {best_feature_size}")
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    print("\nClassification report for best Neural Network model:")
    print(classification_report(
        best_y_test,
        best_prediction,
        target_names=class_names
    ))

    # 3. Confusion matrix for best Neural Network model
    labels = np.arange(len(class_names))
    cm = confusion_matrix(best_y_test, best_prediction, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"Neural Network Confusion Matrix ({best_feature_size} MFCC features)")
    plt.tight_layout()
    plt.savefig(f"neural_network_confusion_matrix_{best_feature_size}.png", dpi=300, bbox_inches="tight")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # 4. Plot training metrics for the best model (Defensive check against missing curve attributes)
    mlp_model = best_model.named_steps["mlp"]
    plt.figure(figsize=(9, 5))
    
    # Check if loss curve data is present
    if hasattr(mlp_model, "loss_curve_") and len(mlp_model.loss_curve_) > 0:
        plt.plot(mlp_model.loss_curve_, label="Training Loss", color="crimson", lw=2)
        
    # Since early_stopping=True, we should also plot the validation accuracy curve if available
    if hasattr(mlp_model, "validation_scores_") and len(mlp_model.validation_scores_) > 0:
        plt.plot(mlp_model.validation_scores_, label="Validation Accuracy", color="forestgreen", lw=2, linestyle="--")

    plt.title(f"Neural Network training metrics over time ({best_feature_size} MFCC features)", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    
    plt.savefig(f"neural_network_metrics_curve_{best_feature_size}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

else:
    print("No valid feature files were processed. Exiting pipeline.")