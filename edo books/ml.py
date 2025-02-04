import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Read the CSV file with ';' as the delimiter
df = pd.read_csv(r"C:\\Users\\rezgu\\OneDrive\\Desktop\\ml\\edo books\\Zomato-data-.csv", encoding='ISO-8859-1', delimiter=';')

# Check the first few rows to understand the data structure
print(df.head())

# Define a function to clean the ratings by splitting on '/' and keeping the first part (the numeric value)
def clean_ratings(value):
    if isinstance(value, str) and '/' in value:
        return float(value.split('/')[0])  # Split on '/' and take the first part (before the '/')
    return value  # If not a string or no '/' found, return the value as it is

# Apply the cleaning function to the 'rate' column (replace 'rate' with your actual column name if different)
df['rate'] = df['rate'].apply(clean_ratings)

# Alternatively, if you want to clean multiple columns, you can loop through the relevant columns:
columns_to_clean = ['rate']  # Add any other columns that need cleaning
for col in columns_to_clean:
    df[col] = df[col].apply(clean_ratings)

# Define a mapping for the categorical values
type_mapping = {
    'Dining': -1,
    'Coffee': 0,
    'Buffet': 1,
    'Other': 2
}

# Apply the mapping to convert categorical values to numerical
df['listed_in(type)'] = df['listed_in(type)'].map(type_mapping)

# Print unique values to verify
print(df['listed_in(type)'].unique())


# Check the cleaned data
print(df.head())

X = df[[ "online_order", "book_table", "votes", "approx_cost(for two people)", "listed_in(type)"]]
y = df["rate"]

# Print the target variable 'y' to check the output
print(y)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_relationships(df, target_column):
    """
    Plots the relationship between all columns in the DataFrame and the specified target column.
    
    - Numerical columns: Scatter plots
    - Categorical columns: Box plots
    """
    # Identify numerical and categorical columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    # Create subplots for numerical features
    plt.figure(figsize=(12, len(numerical_features) * 4))
    for i, feature in enumerate(numerical_features):
        if feature != target_column:  # Avoid plotting rate vs. rate
            plt.subplot(len(numerical_features), 1, i + 1)
            sns.scatterplot(data=df, x=feature, y=target_column, alpha=0.6)
            plt.xlabel(feature)
            plt.ylabel(target_column)
            plt.title(f"{feature} vs {target_column}")

    plt.tight_layout()
    plt.show()

    # Create subplots for categorical features
    plt.figure(figsize=(12, len(categorical_features) * 4))
    for i, feature in enumerate(categorical_features):
        plt.subplot(len(categorical_features), 1, i + 1)
        sns.boxplot(data=df, x=feature, y=target_column)


plot_feature_relationships(df, 'rate')


# Drop rows with missing values
df = df.dropna()  # Or use df.fillna(df.median()) to fill missing values

# Define features and target
X = df[["online_order", "book_table", "votes", "approx_cost(for two people)", "listed_in(type)"]]
y = df["rate"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
}

# Train and evaluate each model (using regression metrics)
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test data
    mse = mean_squared_error(y_test, y_pred)  # Compute mean squared error (MSE)
    print(f"{name} Mean Squared Error: {mse:.2f}")

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the model
rf = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees
    'max_depth': [10, 20, 30],  # Depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples to split node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples to be at a leaf
}

# Apply Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
print(f"Best hyperparameters: {grid_search.best_params_}")

# Best model with optimized hyperparameters
best_rf_model = grid_search.best_estimator_

# Evaluate the tuned model
y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Tuned Random Forest Mean Squared Error: {mse:.2f}")


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define a pipeline with StandardScaler and Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalize features
    ('regressor', LinearRegression())  # Linear Regression model
])

# Define the hyperparameter grid for tuning
param_grid = {
    'regressor__fit_intercept': [True, False]  # Try both with and without intercept
}

# Apply Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best hyperparameters from Grid Search
print(f"Best hyperparameters: {grid_search.best_params_}")

# Best model with optimized hyperparameters
best_lr_model = grid_search.best_estimator_

# Evaluate the tuned model
y_pred = best_lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Tuned Linear Regression Mean Squared Error: {mse:.2f}")

print(y_pred)
print(y_test)