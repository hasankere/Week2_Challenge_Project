import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
#read file from location  
file_path = "C:\\Users\\Hasan\\Desktop\\data science folder\\MachineLearningRating_v3.txt"
# Read the file (adjust delimiter as necessary)
df = pd.read_csv(file_path, delimiter='|')

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
# Fill missing values for numeric columns with mean
for column in df.select_dtypes(include=['float64', 'int64']):
    df[column] = df[column].fillna(df[column].mean())

# Fill missing values for categorical columns with mode
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].fillna(df[column].mode()[0])

# Verify after filling
print(df.isnull().sum())
# Select numerical features
numerical_columns = ['TotalPremium', 'TotalClaims']  # Add other numerical features as needed
numerical_data = df[numerical_columns]

# Ensure no missing values in numerical columns
numerical_data = numerical_data.fillna(0)

# Descriptive statistics for numerical features
print("\nDescriptive Statistics:")
print(numerical_data.describe())

# Variability Measures
for col in numerical_columns:
    print(f"\nVariability Measures for {col}:")
    variance = numerical_data[col].var()
    std_dev = numerical_data[col].std()
    data_range = numerical_data[col].max() - numerical_data[col].min()
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Range: {data_range:.2f}")
# Check for categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
print("\nCategorical Variables:")
print(categorical_columns)

# Check for date columns (if any column contains date values)
date_columns = []
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        date_columns.append(col)
    elif pd.api.types.is_object_dtype(df[col]):
        try:
            pd.to_datetime(df[col])
            date_columns.append(col)
        except ValueError:
            pass
print("\nDate Columns:")
print(date_columns)

# Check sample data to confirm formatting
print("\nSample Data:")
# Check unique values in categorical variables
print("\nUnique Values in Categorical Variables:")
for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} unique values")
#Univariate Analysis to visualize the distributions of numerical and categorical variables
# Select numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Handle missing values for visualization
df[numerical_columns] = df[numerical_columns].fillna(0)
df[categorical_columns] = df[categorical_columns].fillna('Missing')

# Plot histograms for numerical columns
def plot_numerical_distribution(df, numerical_columns):
    plt.figure(figsize=(15, len(numerical_columns) * 4))
    for i, col in enumerate(numerical_columns):
        plt.subplot(len(numerical_columns), 1, i + 1)
        sns.histplot(df[col], kde=True, color='skyblue', bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
# Handle missing values for TotalPremium and TotalClaims
df['TotalPremium'] = df['TotalPremium'].fillna(0)
df['TotalClaims'] = df['TotalClaims'].fillna(0)

# Ensure ZipCode is treated as categorical
df['PostalCode'] = df['PostalCode'].fillna('Unknown')

# Group data by ZipCode and calculate monthly changes
df_grouped = df.groupby('PostalCode')[['TotalPremium', 'TotalClaims']].sum().reset_index()

# Scatter plot: TotalPremium vs TotalClaims by ZipCode
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_grouped, x='TotalPremium', y='TotalClaims', hue='PostalCode', palette='tab10')
plt.title("Relationship Between TotalPremium and TotalClaims by ZipCode")
plt.xlabel("TotalPremium")
plt.ylabel("TotalClaims")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="ZipCode")
plt.tight_layout()
plt.show()
# Calculate correlation matrix
correlation_matrix = df[['TotalPremium', 'TotalClaims']].corr()

# Heatmap for correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Correlation Matrix for TotalPremium and TotalClaims")
plt.show()
# Example: Assuming 'CalculatedPremiumPerTerm' represents risk and 'Province' is the column with provinces
provinces = df['Province'].unique()

# Perform ANOVA (assumes 'CalculatedPremiumPerTerm' represents risk)
risk_by_province = [df[df['Province'] == province]['CalculatedPremiumPerTerm'] for province in provinces]
f_stat, p_value = stats.f_oneway(*risk_by_province)

print(f'ANOVA Results - F-statistic: {f_stat}, p-value: {p_value}')

# Accept or reject the null hypothesis
if p_value < 0.05:
    print("Reject the null hypothesis: There are significant risk differences across provinces.")
else:
    print("Accept the null hypothesis: There are no significant risk differences across provinces.")
    # Example: Assuming 'CalculatedPremiumPerTerm' represents risk and 'PostalCode' represents zip codes
zipcodes = df['PostalCode'].unique()

# Perform ANOVA for postal codes (assuming more than two zip codes are available)
risk_by_zipcode = [df[df['PostalCode'] == zipcode]['CalculatedPremiumPerTerm'] for zipcode in zipcodes]
f_stat, p_value = stats.f_oneway(*risk_by_zipcode)

print(f'ANOVA Results - F-statistic: {f_stat}, p-value: {p_value}')

# Accept or reject the null hypothesis
if p_value < 0.05:
    print("Reject the null hypothesis: There are significant risk differences between zip codes.")
else:
    print("Accept the null hypothesis: There are no significant risk differences between zip codes.")
# Example: Assuming 'TotalPremium' represents the margin/profit and 'PostalCode' represents zip codes
zipcodes = df['PostalCode'].unique()

# Perform ANOVA for postal codes
profit_by_zipcode = [df[df['PostalCode'] == zipcode]['TotalPremium'] for zipcode in zipcodes]
f_stat, p_value = stats.f_oneway(*profit_by_zipcode)

print(f'ANOVA Results - F-statistic: {f_stat}, p-value: {p_value}')

# Accept or reject the null hypothesis
if p_value < 0.05:
    print("Reject the null hypothesis: There are significant margin differences between zip codes.")
else:
    print("Accept the null hypothesis: There are no significant margin differences between zip codes.")
    # Example: Assuming 'CalculatedPremiumPerTerm' represents risk and 'Gender' represents gender
women_risk = df[df['Gender'] == 'Female']['CalculatedPremiumPerTerm']
men_risk = df[df['Gender'] == 'Male']['CalculatedPremiumPerTerm']

# Perform t-test comparing women vs. men
t_stat, p_value = stats.ttest_ind(women_risk, men_risk)

print(f'T-test Results - t-statistic: {t_stat}, p-value: {p_value}')

# Accept or reject the null hypothesis
if p_value < 0.05:
    print("Reject the null hypothesis: There are significant risk differences between women and men.")
else:
    print("Accept the null hypothesis: There are no significant risk differences between women and men.")
    # Relevant features
features = ['Province', 'PostalCode', 'VehicleType', 'make', 'Model', 'Cylinders', 'cubiccapacity', 'SumInsured', 'CalculatedPremiumPerTerm']
target = 'TotalClaims'

# Remove rows with missing values in the selected columns
df_clean = df.dropna(subset=features + [target])

# Group by PostalCode and fit a model for each
zipcodes = df_clean['PostalCode'].unique()
zipcode_models = {}

for zipcode in zipcodes:
    zipcode_data = df_clean[df_clean['PostalCode'] == zipcode]
    
    # Check if the number of samples is sufficient for train-test split
    if len(zipcode_data) < 2:
        print(f"Skipping zipcode {zipcode} due to insufficient data.")
        continue  # Skip this zipcode
    
    # Define features (X) and target (y)
    X = zipcode_data[features]
    y = zipcode_data[target]
    # One-hot encode categorical columns (like VehicleType, make, etc.)
    X = pd.get_dummies(X, drop_first=True)
    
    # Train-Test Split for each zipcode's dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Store the model and its error
    zipcode_models[zipcode] = {'model': model, 'mse': mse}

    print(f"Zipcode {zipcode} - MSE: {mse}")
from scipy.stats import ttest_ind, chi2_contingency
# Select the feature for segmentation (e.g., "CoverType")
feature_to_test = "CoverType"

# Split into Group A (Control) and Group B (Test)
group_a = df[df[feature_to_test] == "Basic"]  # Control: Baseline feature
group_b = df[df[feature_to_test] == "Comprehensive"]  # Test: Feature to test

# Validate group similarity (Numerical Columns)
numerical_columns = ['TotalPremium', 'TotalClaims', 'SumInsured']
for col in numerical_columns:
    stat, p_value = ttest_ind(group_a[col].dropna(), group_b[col].dropna())
    print(f"T-test for {col}: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(f"WARNING: {col} is significantly different between groups (p < 0.05).")
    else:
        print(f"{col} is not significantly different between groups (p >= 0.05).")

# Validate group similarity (Categorical Columns)
categorical_columns = ['Gender', 'Province', 'VehicleType']
for col in categorical_columns:
    contingency_table = pd.crosstab(df[col], df[feature_to_test])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-squared test for {col}: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(f"WARNING: {col} is significantly different between groups (p < 0.05).")
    else:
        print(f"{col} is not significantly different between groups (p >= 0.05).")
        import scipy.stats as stats
#Chi-Squared Test (for Categorical Variables)
# Create a contingency table for 'Gender' and 'IsVATRegistered'
contingency_table = pd.crosstab(df['Gender'], df['IsVATRegistered'])

# Perform the chi-squared test
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# Output results
print(f"Chi-Squared Statistic: {chi2_stat}")
print(f"P-Value: {p_val}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")

# Interpretation
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant relationship between Gender and VAT registration status.")
else:
    print("Fail to reject the null hypothesis: There is no significant relationship between Gender and VAT registration status.")
    #T-Test (for Numerical Variables)
# Split the data by 'Gender' and extract 'TotalPremium' for each group
group_male = df[df['Gender'] == 'Male']['TotalPremium'].dropna()
group_female = df[df['Gender'] == 'Female']['TotalPremium'].dropna()

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(group_male, group_female)

# Output results
print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference in TotalPremium between males and females.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in TotalPremium between males and females.")
    #Z-Test (for Large Samples or Known Population Variance)
# Example: Testing if the mean TotalClaims is significantly different from a population mean of 50
population_mean = 50
sample_mean = df['TotalClaims'].mean()
population_std = df['TotalClaims'].std()  # Assuming you know the population standard deviation
sample_size = len(df['TotalClaims'])

# Calculate the standard error
standard_error = population_std / (sample_size ** 0.5)

# Calculate the z-statistic
z_stat = (sample_mean - population_mean) / standard_error

# Calculate the p-value for two-tailed test
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Output results
print(f"Z-Statistic: {z_stat}")
print(f"P-Value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: The mean TotalClaims is significantly different from the population mean.")
else:
    print("Fail to reject the null hypothesis: The mean TotalClaims is not significantly different from the population mean.")