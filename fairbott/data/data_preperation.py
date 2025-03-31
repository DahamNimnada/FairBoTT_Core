import pandas as pd

class DataPreparation:

    def __init__(self, bias_data_path, max_rows=2000):
        self.bias_data_path = bias_data_path
        self.max_rows = max_rows
        self.data = None

    def load_data(self):
        # Load the dataset
        self.data = pd.read_csv(self.bias_data_path, nrows=self.max_rows)
        print("Data loaded successfully.")

    def drop_missing_values(self):
        # Drop rows where 'text' column is missing
        initial_rows = len(self.data)
        self.data.dropna(subset=['text'], inplace=True)
        final_rows = len(self.data)
        print(f" Dropped {initial_rows - final_rows} rows with missing 'text' values.")

    def explore_data(self):
        # Explore the first few rows of the dataset
        print("\n Bias Data Preview:")
        print(self.data.head())

    def check_missing_values(self):
        # Check for any missing values in the dataset
        print("\n Missing values:")
        print(self.data.isnull().sum())

    def create_labels(self):
        # Updated 19-class label mapping
        bias_type_mapping = {
            'ability': 0,
            'body_type': 1,
            'characteristics': 2,
            'cultural': 3,
            'disabled': 4,
            'gender': 5,
            'nationality': 6,
            'neutral': 7,
            'political_ideologies': 8,
            'profession': 9,
            'race': 10,
            'race_ethnicity': 11,
            'religion': 12,
            'social': 13,
            'socioeconomic_class': 14,
            'unknown': 15,
            'victim': 16,
            'bias_general': 17,
            'other': 18
        }

        # Convert bias_type to numerical label
        self.data['label'] = self.data['bias_type'].map(bias_type_mapping)

        # Assign "unknown" (15) to any unmapped labels
        self.data['label'] = self.data['label'].fillna(bias_type_mapping['unknown']).astype(int)

        print(" Label distribution:\n", self.data['label'].value_counts())

    def get_data(self):
        # Return text and label lists
        return self.data['text'].tolist(), self.data['label'].tolist()