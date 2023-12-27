import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load CSV data
df = pd.read_csv("quality-hr-ann.csv")

# Handle missing values
df = df.dropna()

# Normalize features
scaler = StandardScaler()
df["HR"] = scaler.fit_transform(df[["HR"]])

# Split data into features (X) and target variable (y)
X = df[["HR"]]
y = df["Quality"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kernel Experimentation
kernels = ["linear", "rbf", "poly", "sigmoid"]
best_accuracy = 0
best_kernel = None
best_model = None

for kernel in kernels:
    # Create and train the SVM model
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with kernel '{kernel}':", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel
        best_model = model

print("\nBest kernel:", best_kernel)
print("Best accuracy:", best_accuracy)

# Visualize decision boundary (using the best model)
plt.scatter(X_test["HR"], y_test, c=y_test, cmap="viridis")
plt.plot(X_test["HR"], best_model.predict(X_test), color="red", linewidth=2, label="Predicted")
plt.xlabel("Heart Rate")
plt.ylabel("Quality")
plt.title("SVM Decision Boundary (Best Kernel)")
plt.legend()
plt.show()

# Create confusion matrix (using the best model)
cm = confusion_matrix(y_test, best_model.predict(X_test))
print("Confusion Matrix:\n", cm)
