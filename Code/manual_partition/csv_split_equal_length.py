import pandas as pd

# Load the original dataset
file_path = './data/3/new_train_set6.csv'
df = pd.read_csv(file_path)

# Define the line number ranges (1-based, inclusive, assuming line 1 is the header)
part_ranges = {
    'upper_front_buccal': (237, 926),
    'upper_left_buccal': (1144, 2012),
    'upper_right_buccal': (2322, 3308),
    'upper_left_occlusal': (3671, 4977),
    'upper_right_occlusal': (5157, 5880),
    'upper_left_lingual': (6064, 6799),
    'upper_front_lingual': (6899, 7621),
    'upper_right_lingual': (7744, 9131)
}

# Calculate the minimum number of rows across all specified ranges
# (end - start + 1) gives the total inclusive rows for that range
min_rows = min((end - start + 1) for start, end in part_ranges.values())
print(f"Targeting a uniform length of {min_rows} rows for all files.\n")

# Extract, trim, and save each part
for part, (start_line, _) in part_ranges.items():
    # Convert 1-based line numbers to 0-based pandas indices
    # Subtract 2 because: Line 1 is the header, Line 2 is index 0
    start_idx = start_line - 2
    
    # Calculate the new end index by adding the minimum row count
    end_idx = start_idx + min_rows
    
    # Slice the dataframe to the uniform length
    df_sliced = df.iloc[start_idx:end_idx]
    
    # Construct filename and save to CSV
    output_filename = f"{part}_s6.csv"
    df_sliced.to_csv(output_filename, index=False)
    print(f"Saved {output_filename} with {len(df_sliced)} rows.")

print("\nAll files have been successfully separated and uniformly sized.")