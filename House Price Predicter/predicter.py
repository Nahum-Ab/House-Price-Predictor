import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt # To visualize the prediction of the model vs actual price

# Load The Dataset
data = pd.read_csv('train.csv')

# Preview the dataset to identify features
print(data.head())

# Select relevant features for the model
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']

# Handle missing values
features = features.dropna()
target = target[features.index]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using Mean Squared Error [summation((actualPrice - PredictedPrice)^2)/n]
mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)


# print(predictions)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(Y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()