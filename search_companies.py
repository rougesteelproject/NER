import pickle
import os
import string
import re

# Define a set of common English stop words
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "not", "for", "on", "in", "at", "with",
    "as", "by", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "whose", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "should", "could",
    "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've",
    "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
    "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't",
    "mustn't", "let's", "that's", "who's", "what's", "here's", "there's", "when's",
    "where's", "why's", "how's", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
}

def clean_name(name):
    """Converts to lowercase, strips whitespace, and removes punctuation for consistent matching."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = name.lower()
    # Remove punctuation, but keep spaces for multi-word phrases
    name = re.sub(r'[^\w\s]', '', name)
    return name

class CompanyNameSearcher:
    def __init__(self, tries_dir="aho_corasick_tries"):
        self.tries_with_ranges = [] # Store tuples of (trie, min_name, max_name)
        self.load_tries(tries_dir)

    def load_tries(self, tries_dir):
        """Loads all pickled Aho-Corasick tries and their ranges from the specified directory."""
        if not os.path.exists(tries_dir):
            print(f"Error: Directory '{tries_dir}' not found.")
            return

        print(f"Loading Aho-Corasick tries from '{tries_dir}'...")
        for filename in sorted(os.listdir(tries_dir)):
            if filename.endswith(".pkl"):
                filepath = os.path.join(tries_dir, filename)
                with open(filepath, 'rb') as f:
                    trie_data = pickle.load(f)
                    trie = trie_data['trie']
                    min_name, max_name = trie_data['range']
                    self.tries_with_ranges.append((trie, min_name, max_name))
                print(f"Loaded {filename} (Range: {min_name} - {max_name})")
        print(f"Finished loading {len(self.tries_with_ranges)} tries.")

    def find_company_names(self, text):
        """
        Searches the given text for company names using the loaded tries.
        Generates all possible whole word and multi-word phrases from the input text
        and checks each phrase against the tries for exact matches.
        """
        found_names = set()
        cleaned_text = clean_name(text)
        print(f"DEBUG: Cleaned input text: '{cleaned_text}'")
        
        if not cleaned_text:
            return []

        # Iterate through all tries and search the cleaned text
        for trie, _, _ in self.tries_with_ranges:
            for end_index, (original_name, cleaned_keyword) in trie.iter(cleaned_text):
                start_index = end_index - len(cleaned_keyword) + 1
                
                # Check for word boundaries using regex for more robust detection
                # We need to ensure the matched keyword is a standalone word/phrase
                # by checking characters immediately before and after the match.
                # \b in regex matches word boundaries.
                # However, Aho-Corasick's iter gives us end_index, so we construct the check manually.
                
                # Check if the character before the match is a word boundary (non-alphanumeric or start of string)
                pre_match_char = cleaned_text[start_index - 1] if start_index > 0 else ' '
                is_start_boundary = not pre_match_char.isalnum()
                
                # Check if the character after the match is a word boundary (non-alphanumeric or end of string)
                post_match_char = cleaned_text[end_index + 1] if end_index < len(cleaned_text) - 1 else ' '
                is_end_boundary = not post_match_char.isalnum()

                if is_start_boundary and is_end_boundary:
                    # Check if the cleaned_keyword is a single stop word
                    if ' ' not in cleaned_keyword and cleaned_keyword in STOP_WORDS:
                        print(f"DEBUG: Matched stop word '{cleaned_keyword}', skipping.")
                        continue
                    
                    found_names.add(original_name)
                    print(f"DEBUG: Found whole word match for '{cleaned_keyword}' -> Original Value: '{original_name}'")
                else:
                    print(f"DEBUG: Found partial match for '{cleaned_keyword}' (not whole word) in '{cleaned_text}'")
        
        return list(found_names)

if __name__ == "__main__":
    # Example usage:
    searcher = CompanyNameSearcher()
    
    if searcher.tries_with_ranges:
        test_text1 = "This text mentions Google and Microsoft, but not Apple."
        found1 = searcher.find_company_names(test_text1)
        print(f"\nSearching text: '{test_text1}'")
        print(f"Found company names: {found1}")

        test_text2 = "Our client is Amazon, a big tech company."
        found2 = searcher.find_company_names(test_text2)
        print(f"\nSearching text: '{test_text2}'")
        print(f"Found company names: {found2}")

        test_text3 = "I work for 'Acme Corp' and 'XYZ Inc.'."
        found3 = searcher.find_company_names(test_text3)
        print(f"\nSearching text: '{test_text3}'")
        print(f"Found company names: {found3}")
    else:
        print("\nNo tries loaded. Please ensure 'aho_corasick_tries' directory exists and contains pickled tries.")