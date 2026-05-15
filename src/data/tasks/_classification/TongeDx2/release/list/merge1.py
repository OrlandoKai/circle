import pandas as pd

# Define the unified column order
columns = [
    'id', 'Gender', 'Age', 'Pulse', 'TonguePale', 'TipSideRed', 'Spot',
    'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow',
    'Age_Group', 'image_path', 'Heart', 'Lung', 'Spleen', 'Liver', 'Kidney'
]

# Load train_fold1.csv
train_df = pd.read_csv('train_fold1.csv')
# Add missing Pulse column
train_df['Pulse'] = ''
# Reorder columns
train_df = train_df[columns]

# Load val_fold1.csv
val_df = pd.read_csv('val_fold1.csv')
# Add missing Pulse column
val_df['Pulse'] = ''
# Reorder columns
val_df = val_df[columns]

# Load test.csv
test_df = pd.read_csv('test.csv')
# Reorder columns to match
test_df = test_df[columns]

# Concatenate all DataFrames
merged_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Sort by id
merged_df['id'] = merged_df['id'].astype(int)
merged_df = merged_df.sort_values(by='id').reset_index(drop=True)

# Save to new CSV
merged_df.to_csv('merged_data.csv', index=False)

print("Merged CSV saved as merged_data.csv")
