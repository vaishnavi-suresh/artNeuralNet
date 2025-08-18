import pandas as pd

# Read the artists.csv file
artists_df = pd.read_csv('./files/artists.csv')

# Prepare a list to hold the new rows
new_rows = []

# Iterate through each row of the DataFrame
for idx, row in artists_df.iterrows():
    artist_name = row[1]  # Assuming artist name is in the first column
    paintings_count = int(row[7])  # 'paintings' column
    genre = row[3]  # Column 3 (0-based index)
    # Format artist name as first_last
    name_parts = artist_name.strip().split(' ')
    first_last = '_'.join(name_parts)
    for i in range(paintings_count):
        image_name = f"{first_last}_{i}"
        new_rows.append({
            'image_name': image_name,
            'genre': genre
        })

# Create a new DataFrame from the new rows
output_df = pd.DataFrame(new_rows)

# Save to a new CSV file
output_df.to_csv('./files/artist_images.csv', index=False)
