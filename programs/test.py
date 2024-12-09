import pandas as pd

def analyze_score_and_count_distribution(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print("Columns in the file:", df.columns)  # Debugging step
    
    # Ensure column names are clean
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    
    # Convert 'score' column to numeric, handling errors
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score'])  # Drop rows with NaN in 'score'
    
    # Calculate score distribution
    score_bins = [1, 2, 3, 4, 5, 6]
    score_labels = ['1-2', '2-3', '3-4', '4-5', '5-6']
    df['score_bin'] = pd.cut(df['score'], bins=score_bins, labels=score_labels, right=False)
    score_distribution = df['score_bin'].value_counts().sort_index()

    # Convert 'count' column to numeric and handle errors
    df['count'] = pd.to_numeric(df['count'], errors='coerce')  # Ensure 'count' is numeric
    df['count_group'] = df['count'].apply(lambda x: x if x < 6 else '6+')
    
    # Convert 'count_group' to string to avoid sorting issues
    df['count_group'] = df['count_group'].astype(str)
    count_distribution = df['count_group'].value_counts().sort_index()

    # Print results
    print("Score Distribution:")
    for label, count in score_distribution.items():
        print(f"{label}: {count}")

    print("\nCount Distribution:")
    for label, count in count_distribution.items():
        print(f"{label}: {count}")

    return score_distribution, count_distribution

# Replace 'file_path.csv' with the actual path to your CSV file
file_path = '../../data/data.csv'
analyze_score_and_count_distribution(file_path)
