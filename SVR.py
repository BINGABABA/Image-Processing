import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error

# Load CSV data
df = pd.read_csv("quality-hr-ann.csv")

# Handle missing values
df = df.dropna()

# Normalize features
scaler = StandardScaler()
df["HR"] = scaler.fit_transform(df[["HR"]])

# Split data into training and testing sets
X = df[["HR"]]
y = df["Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the kernels to try
kernels = ["linear", "rbf", "poly", "sigmoid"]  # Define the kernels list

# Define hyperparameter grids for different kernels
param_grids = {
    "linear": {"C": [0.1, 1, 10, 100]},
    "rbf": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "poly": {"C": [0.1, 1, 10, 100], "degree": [2, 3, 4]},
    "sigmoid": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
}

# ... (rest of the code) ...

# Iterate through kernels and perform hyperparameter tuning
for kernel in kernels:
    print("\nKernel:", kernel)

    # Create grid search object
    grid_search = GridSearchCV(SVR(kernel=kernel), param_grid=param_grids[kernel], cv=5)

    # Perform grid search to find best hyperparameters
    grid_search.fit(X_train, y_train)

    # Get best model and its hyperparameters
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    # Use the tuned model for evaluation and plotting
    y_pred = best_model.predict(X_test)
    y_pred_rounded = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, y_pred_rounded)
    mse = mean_squared_error(y_test, y_pred_rounded)
    print("Accuracy:", accuracy)
    print("Mean Squared Error (5 decimal places):", round(mse, 5))

    # Plot predictions vs. actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test["HR"], y_test, alpha=0.6, label="Actual")
    plt.plot(X_test["HR"], y_pred_rounded, color="red", linewidth=2, label="Predicted ({})".format(kernel))
    plt.xlabel("Heart Rate")
    plt.ylabel("Quality")
    plt.title("SVR Predictions vs. Actual (Kernel: {})".format(kernel))
    plt.legend()
    plt.show()
