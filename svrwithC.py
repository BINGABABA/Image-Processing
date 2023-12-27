import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Create and train the SVR model with hyperparameter tuning
model = SVR(kernel="rbf",  # Experiment with different kernels
            C=20,  # Adjust regularization parameter
            gamma="scale")  # Consider different gamma values
model.fit(X_train, y_train)



# Make predictions on the testing set
y_pred = model.predict(X_test)
y_pred_rounded = [round(value) for value in y_pred]  # Round predictions to integers
print("Classification Report (with zero_division='warn'):\n", classification_report(y_test, y_pred_rounded, zero_division="warn"))
# Evaluate model performance with detailed metrics
accuracy = accuracy_score(y_test, y_pred_rounded)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rounded))
print("Classification Report:\n", classification_report(y_test, y_pred_rounded))
