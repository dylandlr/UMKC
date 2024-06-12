import pandas as pd

# Load the CSV file from the provided path
file_path = 'model_performance_comparison.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand its structure
print(data.head())

# Calculate the accuracy of the model
accuracy = (data['actual'] == data['predicted']).mean()

print(f"The accuracy of the model is: {accuracy * 100:.2f}%")
