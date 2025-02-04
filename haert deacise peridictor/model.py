import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import resample
df = pd.read_csv(r"C:\\Users\\rezgu\\OneDrive\\Desktop\\ml\\haert deacise peridictor\\Heart_disease_statlog.csv")



df.columns = ['age', 'sex', 'chestpain','restbpressure','chol','fastingbloodsugar','restecg','highestbloodrates','Angina','oldpeak','slope','major_vessels','thalassemia','Target']

print(df.head())

df.dropna()

print(df['Target'].value_counts())

# Separate the classes
df_target_0 = df[df['Target'] == 0]
df_target_1 = df[df['Target'] == 1]


# Select X (all columns except 'target')
X = df.drop('Target', axis=1)

# Select y (only 'target' column)
y = df['Target']

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Ploting"""
import seaborn as sns 


# Scale features (important for models like SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
}

# Train each model and evaluate
results = {}

for name, model in models.items():
    # Fit the model on the training data
    model.fit(X_train_scaled, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(name)
    print(y_test)
    print(y_pred)
    results[name] = accuracy

# Create a DataFrame to compare model performances
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

# Sort results by accuracy
results_df = results_df.sort_values(by='Accuracy', ascending=False)

# Print comparison table
print(results_df)

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison')
plt.show()

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(LogisticRegression(), X, y, cv=5, scoring='accuracy').mean()
print(f"Cross-validated Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(auc)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")
