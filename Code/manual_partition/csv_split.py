import pandas as pd

# Load the original dataset
# Ensure this script is in the same directory as 'new_train_set5.csv'
file_path = './data/3/new_train_set7.csv'
df = pd.read_csv(file_path)

# Define the line number ranges (1-based, inclusive, assuming line 1 is the header)
# The keys correspond to the desired output filenames. 
# Note: Added 'upper_' to front_buccal based on your example. 
# You can easily add 'upper_' or 'lower_' to the other keys if needed for your pipeline.
part_ranges = {
    'upper_front_buccal': (368, 1411),
    'upper_left_buccal': (1562, 2705),
    'upper_right_buccal': (2723, 3451),
    'upper_left_occlusal': (3646, 5021),
    'upper_right_occlusal': (5030, 6001),
    'upper_left_lingual': (6218, 7145),
    'upper_front_lingual': (7251, 7954),
    'upper_right_lingual': (8179, 9079)
}

# Extract and save each part
for part, (start_line, end_line) in part_ranges.items():
    # Convert 1-based line numbers to 0-based pandas indices
    # Subtract 2 because: Line 1 is the header, Line 2 is index 0
    start_idx = start_line - 2
    end_idx = end_line - 1  # end_line - 2 for 0-index, +1 to make it inclusive in the slice
    
    # Slice the dataframe
    df_sliced = df.iloc[start_idx:end_idx]
    
    # Construct filename and save to CSV, keeping the same format as the original
    output_filename = f"{part}_s7.csv"
    df_sliced.to_csv(output_filename, index=False)
    print(f"Saved {output_filename} with {len(df_sliced)} rows.")

print("All files have been successfully separated and saved.")