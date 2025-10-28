import polars as pl
import ahocorasick
import pickle
import os
import string

def clean_name(name):
    """Converts to lowercase, strips whitespace, and removes punctuation for consistent matching."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = name.lower()
    # Remove punctuation
    name = name.translate(str.maketrans('', '', string.punctuation))
    return name

def create_and_pickle_tries(csv_path, output_dir="aho_corasick_tries", chunk_size=100000):
    """
    Reads company names from a CSV, splits them into chunks, builds Aho-Corasick tries,
    and pickles each trie along with its alphabetical range to a separate file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading company names from {csv_path}...")
    
    # Read the 'name' column in chunks using Polars
    # Polars does not have a direct 'chunksize' for CSV reading like pandas.
    # We'll read the entire column and then process it in chunks.
    # For very large files, consider reading line by line or using a custom chunking logic if memory is still an issue.
    df = pl.read_csv(csv_path, columns=['name'])
    
    print("Cleaning and sorting company names using Polars...")
    # Apply cleaning function and create a new 'cleaned_name' column
    # Also handle potential non-string values by casting to Utf8
    df = df.with_columns(
        pl.col("name").cast(pl.Utf8).map_elements(lambda x: clean_name(x) if x is not None else "").alias("cleaned_name")
    )
    
    # Filter out empty cleaned names and sort by 'cleaned_name'
    df = df.filter(pl.col("cleaned_name") != "").sort("cleaned_name")
    
    # Convert to a list of tuples (cleaned_name, original_name)
    processed_names = df.select([pl.col("cleaned_name"), pl.col("name")]).rows()
    print("Cleaning and sorting complete.")

    trie_count = 0
    for i in range(0, len(processed_names), chunk_size):
        chunk_data = processed_names[i:i + chunk_size]
        A = ahocorasick.Automaton()
        
        cleaned_names_in_chunk = []
        print(f"Processing chunk {trie_count+1} with {len(chunk_data)} company names...")
        for cleaned_name, original_name in chunk_data:
            A.add_word(cleaned_name, (original_name, cleaned_name)) # Store tuple (original_name, cleaned_name)
            cleaned_names_in_chunk.append(cleaned_name)
        
        A.make_automaton()
        
        # Determine alphabetical range for the chunk
        min_name = cleaned_names_in_chunk[0] if cleaned_names_in_chunk else ""
        max_name = cleaned_names_in_chunk[-1] if cleaned_names_in_chunk else ""

        # Store the trie and its metadata
        trie_data = {
            'trie': A,
            'range': (min_name, max_name)
        }
        
        output_file = os.path.join(output_dir, f"trie_chunk_{trie_count+1}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(trie_data, f)
        print(f"Pickled trie chunk {trie_count+1} to {output_file} (Range: {min_name} - {max_name})")
        trie_count += 1
    
    print(f"Finished creating and pickling {trie_count} Aho-Corasick tries.")

if __name__ == "__main__":
    csv_file = "companies_sorted.csv"
    create_and_pickle_tries(csv_file)