import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\\Users\\rezgu\\OneDrive\\Desktop\\machine learning netflix\\netflix_titles.csv")

# Convert 'type' to numerical: 1 for 'Movie', 0 for 'TV Show'
df['type'] = df['type'].apply(lambda x: 1 if x == 'Movie' else 0)

# Separate majority and minority classes
df_movies = df[df['type'] == 1]  
df_tvshows = df[df['type'] == 0]  

# Downsample majority class (Movies) to match TV Shows count
df_movies_downsampled = df_movies.sample(n=len(df_tvshows), random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_movies_downsampled, df_tvshows]).sample(frac=1, random_state=42).reset_index(drop=True)

# Define X (features) and y (target)
X = df_balanced.drop(columns=['type'])  
y = df_balanced['type']  

# Identify numerical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns

# Plot numerical features against y
plt.figure(figsize=(12, len(num_features) * 4))
for i, feature in enumerate(num_features):
    plt.subplot(len(num_features), 2, i + 1)
    plt.scatter(X[feature], y, c=y, cmap="bwr", alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("Type (0 = TV Show, 1 = Movie)")
    plt.title(f"{feature} vs Type")
    
plt.tight_layout()
plt.show()

#not studyiable values are 90% text