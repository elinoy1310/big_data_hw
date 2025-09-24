import re

# List of source files
files = [
    r"big_data_recommendation_system\data\BX-Books_hkv1.sql",
    r"big_data_recommendation_system\data\BX-Users_hkv1.sql",
    r"big_data_recommendation_system\data\BX-Book-Ratings_hkv1.sql",
]

# List of output files for SQLite
output_files = [
    r"big_data_recommendation_system\data\SQLite_BX-Books.sql",
    r"big_data_recommendation_system\data\SQLite_BX-Users.sql",
    r"big_data_recommendation_system\data\SQLite_Ratings.sql",
]

# Number of INSERTs per transaction
batch_size = 25000

for file, output_file in zip(files, output_files):
    insert_count = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                # Remove unsupported MySQL options
                line = re.sub(r'ENGINE=\w+\s*', '', line, flags=re.IGNORECASE)
                line = re.sub(r'DEFAULT CHARSET=\w+\s*', '', line, flags=re.IGNORECASE)

                # Table names
                line = re.sub(r'`BX-Users`', 'BX_Users', line)
                line = re.sub(r'`BX-Books`', 'BX_Books', line)
                line = re.sub(r'`BX-Book-Ratings`', 'BX_Book_Ratings', line)

                # Column names
                line = line.replace('User-ID', 'User_ID')
                line = line.replace('Book-Rating', 'Book_Rating')
                line = line.replace('Year-Of-Publication', 'Year_Of_Publication')
                line = line.replace('Book-Title', 'Book_Title')
                line = line.replace('Book-Author', 'Book_Author')
                line = line.replace('Image-URL-S', 'Image_URL_S')
                line = line.replace('Image-URL-M', 'Image_URL_M')
                line = line.replace('Image-URL-L', 'Image_URL_L')

                # INSERT IGNORE → INSERT OR IGNORE
                line = re.sub(r'INSERT IGNORE INTO', 'INSERT OR IGNORE INTO', line, flags=re.IGNORECASE)
                line = line.replace(r"\'", "''")

                # Add BEGIN TRANSACTION every batch_size INSERTs
                if 'INSERT' in line.upper():
                    if insert_count % batch_size == 0:
                        if insert_count > 0:
                            out_f.write("COMMIT;\n")
                        out_f.write("BEGIN TRANSACTION;\n")
                    insert_count += 1

                out_f.write(line)

        # Final transaction commit
        if insert_count > 0:
            out_f.write("COMMIT;\n")

    print(f"✅ Done! SQLite file ready: {output_file}")
