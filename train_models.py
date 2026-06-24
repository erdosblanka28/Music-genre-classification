import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


## KNN ##

## Data loading
saved_data = np.load("music_features_60.npz")
x = saved_data["features"]
y = saved_data["labels"]

num_features = x.shape[1]

##Data split (80% learn, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)

#model
knn = KNeighborsClassifier(n_neighbors=27) #k=sqrt(n)
knn.fit(x_train,y_train)
knn_prediction = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_prediction)
#print(f"\nA kNN modell pontos eredménye (k=27): {knn_accuracy * 100:.2f}%")

#graph
k_values = list(range(1,51))
accuracies = []

for k in k_values:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(x_train,y_train)
    knn_test_prediction = knn_test.predict(x_test)
    accuracies.append(accuracy_score(y_test, knn_test_prediction))

#searching for best k value
best_index = np.argmax(accuracies)
best_k = k_values[best_index]
best_accuracy = accuracies[best_index]
theoretical_accuracy = accuracies[26]

plt.figure(figsize=(11,6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', markersize=6, label='Measured accuracy')

plt.axvline(x=27, color='r', linestyle='--', linewidth=2, label=f'Theoretical ideal k=27 ($\\sqrt{{n}}$) ({theoretical_accuracy*100:.1f}%)')
plt.axvline(x=best_k, color='g', linestyle='--', linewidth=2, label= f'Measured best k={best_k} ({best_accuracy*100:.1f}%)')

plt.title(f"Accuracy of kNN as a function of neighbours(k)\n (Measured over the {num_features}-dimensional MFCC space)", fontsize=12, fontweight='bold')
plt.xlabel("Number of neighbours(k)", fontsize=10)
plt.ylabel("Accuracy", fontsize=10)
plt.xticks(sorted(list(range(0,51,5))+[27, best_k]))
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

#save figure
plt.savefig(f"knn_k_value_test_{num_features}.png", dpi=300)

plt.show()
