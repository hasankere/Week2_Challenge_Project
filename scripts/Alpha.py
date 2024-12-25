import pandas as pd
#read file from location  
file_path = "C:\\Users\\Hasan\\Desktop\\data science folder\\MachineLearningRating_v3.txt"
# Read the file (adjust delimiter as necessary)
df = pd.read_csv(file_path, delimiter='\t')

 #Set display options
pd.set_option('display.max_rows', 1000)  # Limit the number of rows
pd.set_option('display.max_columns', None)  # Show all columns

# Print the DataFrame
print(df)
if df.isnull().values.any():
    print("The dataset contains missing values.")
else:
    print("No missing values in the dataset.")
    # Calculate missing values
missing_count = df.isnull().sum()  # Count of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100  # Percentage of missing values

# Combine both into a single DataFrame
missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percentage
})

# Display columns with missing data only
missing_summary = missing_summary[missing_summary['Missing Count'] > 0]

print("Missing Values Summary:")
print(missing_summary)