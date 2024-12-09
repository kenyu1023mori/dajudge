import pandas as pd

def remove_duplicates_and_add_column(file_path, output_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['Title'])

    # Remove duplicates
    df = df.drop_duplicates()

    # Add a new column with default value 0
    df['Value'] = 1

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False, header=False)

# Replace 'input.csv' with the path to your input CSV file
# Replace 'output.csv' with the desired path for the output CSV file
file_path = '../../data/sampled_dajare.csv'
output_path = '../../data/sampled_dajare.csv'
remove_duplicates_and_add_column(file_path, output_path)
