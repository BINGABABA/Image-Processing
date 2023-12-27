
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - ((1 - r2) * (len(y) - 1) / (len(y) - X.shape[1]))

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Adjusted R-squared:", adjusted_r2)

# Visualize predictions vs. actual values
plt.scatter(X_test["HR"], y_test, alpha=0.6, label="Actual")
plt.plot(X_test["HR"], y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Heart Rate")
plt.ylabel("Quality")
plt.title("Linear Regression Predictions vs. Actual")
plt.legend()
plt.show()

# Optional: Residuals plot
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel("Predicted Quality")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.show()

