import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load CSV data
df = pd.read_csv("quality-hr-ann.csv")

# Handle missing values
df = df.dropna()

# Split data into features (X) and target variable (y)
X = df.drop("Quality", axis=1)  # Assuming multiple features, adjust accordingly
y = df["Quality"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model with hyperparameter tuning
model = DecisionTreeClassifier(max_depth=4,  # Adjust max_depth as needed
                               min_samples_leaf=5,  # Control overfitting
                               random_state=42)  # Ensure reproducibility
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Visualize the decision tree (optional, for understanding)
 
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 6))
class_names = list(map(str, y.unique()))  # Convert numerical classes to strings
plot_tree(model, filled=True, feature_names=X.columns, class_names=class_names)  # Pass string class names
plt.show()

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
