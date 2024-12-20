# Model-learning
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load dataset and select relevant columns
df = pd.read_csv('spotify_tracks.csv')
data = df[['popularity', 'duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness']]

# Generate and visualize the correlation matrix
correlation_matrix = data.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Perform Simple Linear Regression
X = data[['popularity']]  # Predictor
y = data['danceability']  # Response variable

# Add a constant for the regression model
X = sm.add_constant(X)

# Fit the model
ols_model = sm.OLS(y, X).fit()

# Display the regression summary
print(ols_model.summary())

# Predict values using the regression model
predictions = ols_model.predict(X)

# Visualize the scatter plot with the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X['popularity'], y, alpha=0.6, edgecolor='k', label='Data Points')
plt.plot(X['popularity'], predictions, color='red', linewidth=2, label='Regression Line')
plt.title("Simple Linear Regression: Popularity vs Danceability")
plt.xlabel("Popularity")
plt.ylabel("Danceability")
plt.legend()
plt.show()
