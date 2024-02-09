import pandas as pd

# Define excluded_strings and disease_labels at the module level
excluded_strings = [
    'normal fundus', 'diabetes', 'glaucoma', 'cataract',
    'age related macular degeneration', 'hypertension',
    'pathological myopia', 'other diseases/abnormalities'
]

disease_labels = {
    "moderate non proliferative retinopathy": "MONR",
    "mild nonproliferative retinopathy": "MINR",
    "dry age-related macular degeneration": "DAMD",
    "severe nonproliferative retinopathy": "SNR",
    "drusen": "DR"
}

def clean_encoding_issues(df):
    def replace_commas(disease_string):
        # Replace the non-standard comma with a standard one
        return disease_string.replace('ï¼Œ', ',')
    
    df['Left-Diagnostic Keywords'] = df['Left-Diagnostic Keywords'].apply(replace_commas)
    df['Right-Diagnostic Keywords'] = df['Right-Diagnostic Keywords'].apply(replace_commas)
    return df

def remove_duplicates(df):
    return df.drop_duplicates(subset='ID', keep='first').copy()

def filter_excluded_strings(df, excluded_strings):
    def filter_strings(disease_counts):
        return disease_counts[~disease_counts.index.isin(excluded_strings)]
    
    combined_disease_counts = pd.concat([df['Left-Diagnostic Keywords'], df['Right-Diagnostic Keywords']]).value_counts()
    top_five_combined_excluded = filter_strings(combined_disease_counts)
    print(top_five_combined_excluded)
    return df

def encode_disease_labels(df, disease_labels):
    for label in disease_labels.values():
        df[label] = 0
    for disease, label in disease_labels.items():
        condition = (df['Left-Diagnostic Keywords'].str.contains(disease, case=False, na=False)) | \
                    (df['Right-Diagnostic Keywords'].str.contains(disease, case=False, na=False))
        df[label] = condition.astype(int)
    return df

def encode_patient_sex(df):
    df['Patient Sex'] = df['Patient Sex'].apply(lambda x: 1 if x == 'Female' else 0)
    return df

def preprocess_data(df_path):
    df = pd.read_csv(df_path)
    df = clean_encoding_issues(df)
    df = remove_duplicates(df)
    
    # Directly use the module-level variables
    df = filter_excluded_strings(df, excluded_strings)
    df = encode_disease_labels(df, disease_labels)
    df = encode_patient_sex(df)
    return df
