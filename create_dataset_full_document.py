import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the root directory containing the dataset directories
root_dir = r'C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\full_document_datasets\complete_documents\augmentations'

# Initialize lists to store file paths and labels
file_paths = []
labels = []
directory=[]
# Loop over each subdirectory and gather file paths and labels
for sub_dir in ['1_corners', '2_corners', '3_corners', 'complete_doc']:
    dir_path = os.path.join(root_dir, sub_dir)
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Add other image formats if needed
                file_paths.append(os.path.join(root_dir,sub_dir, file_name))
                labels.append(0 if sub_dir == 'complete_doc' else 1)
                directory.append(sub_dir)
# Create a DataFrame with file paths and labels
df = pd.DataFrame({
    'file_path': file_paths,
    'label': labels,
    "directory":directory
})
# Split the data into train (80%), test (10%), and validation (10%) sets with stratification
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['directory'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['directory'], random_state=42)
#%%
# Save the DataFrames to CSV files
train_df.to_csv(os.path.join(root_dir,'train.csv'), index=False)
val_df.to_csv(os.path.join(root_dir,'val.csv'), index=False)
test_df.to_csv(os.path.join(root_dir,'test.csv'), index=False)
