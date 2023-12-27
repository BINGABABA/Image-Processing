import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
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
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Create and train the SVR model
svr_model = SVR(kernel="rbf")  # Adjust kernel as needed
svr_model.fit(X_train, y_train)

# Make predictions with both models
y_pred_linear = linear_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)

# Evaluate performance
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print("Linear Regression MSE:", mse_linear)
print("Linear Regression R-squared:", r2_linear)
print("SVR MSE:", mse_svr)
print("SVR R-squared:", r2_svr)

# Visualize predictions for both models
plt.figure(figsize=(10, 6))
plt.scatter(X_test["HR"], y_test, alpha=0.6, label="Actual")
plt.plot(X_test["HR"], y_pred_linear, color="blue", linewidth=2, label="Linear Regression")
plt.plot(X_test["HR"], y_pred_svr, color="red", linewidth=2, label="SVR")
plt.xlabel("Heart Rate")
plt.ylabel("Quality")
plt.title("Linear Regression vs. SVR Predictions")
plt.legend()
plt.show()
