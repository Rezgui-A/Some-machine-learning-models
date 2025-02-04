import kagglehub

# Download latest version
path = kagglehub.dataset_download("ruchikakumbhar/zomato-dataset")

print("Path to dataset files:", path)