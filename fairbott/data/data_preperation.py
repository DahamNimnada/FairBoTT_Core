import pandas as pd

class DataPreparation:

    def __init__(self, bias_data_path, max_rows=20000):
        self.bias_data_path = bias_data_path
        self.max_rows = max_rows
        self.data = None

    def load_data(self):
        # Load the dataset
        self.data = pd.read_csv(self.bias_data_path, nrows=self.max_rows)
        print("Data loaded successfully.")

    def drop_missing_values(self):
        # Drop rows where 'full_text' column is missing
        initial_rows = len(self.data)
        self.data.dropna(subset=['full_text', 'bias_label'], inplace=True)
        final_rows = len(self.data)
        print(f"Dropped {initial_rows - final_rows} rows with missing values.")

    def explore_data(self):
        # Preview first few rows
        print("\nBias Data Preview:")
        print(self.data.head())

    def check_missing_values(self):
        # Check missing data in each column
        print("\nMissing values:")
        print(self.data.isnull().sum())

    def get_data(self):
        # Return full_text and bias_label
        return self.data['full_text'].tolist(), self.data['bias_label'].tolist()