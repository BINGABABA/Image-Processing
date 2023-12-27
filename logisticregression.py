import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load CSV data
df = pd.read_csv("quality-hr-ann.csv")

# Handle missing values
df = df.dropna()

# Normalize features
scaler = StandardScaler()
df["HR"] = scaler.fit_transform(df[["HR"]])  # Assuming only one feature, adjust if needed

# Split data into features (X) and target variable (y)
X = df.drop("Quality", axis=1)
y = df["Quality"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(random_state=42)  # Ensure reproducibility
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision boundary (optional, for 1-2 features)
plt.scatter(X_test["HR"], y_test, c=y_test, cmap="viridis")
plt.plot(X_test["HR"], model.predict_proba(X_test)[:, 1], color="red", linewidth=2, label="Predicted Probability")
plt.xlabel("Heart Rate")
plt.ylabel("Quality")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
