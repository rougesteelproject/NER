import spacy
import polars as pl
from search_companies import CompanyNameSearcher

# Load the custom spaCy NER model
try:
    nlp = spacy.load("ProfessionFacilityExperience_LPDoctor/en_core_web_sm_job_related")
except OSError:
    print("Model 'en_core_web_sm_job_related' not found. Please ensure it's installed or the path is correct.")
    print("You might need to run: pip install ProfessionFacilityExperience_LPDoctor/en_core_web_sm_job_related/en_core_web_sm_job-any-py3-none-any.whl")
    exit()

# Initialize CompanyNameSearcher
company_searcher = CompanyNameSearcher()

# Function to extract entities
def extract_entities(text):
    doc = nlp(str(text))
    entities = {
        "PROFESSION": [],
        "FACILITY": [],
        "EXPERIENCE": [],
        "MONEY": []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

# Read the input parquet file
input_file = 'candidate_searches_built_in.parquet'
try:
    df = pl.read_parquet(input_file)
except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
    exit()

# Assuming the text to be processed is in a column named 'text'
# If your column has a different name, please change 'text' below.
if 'text' not in df.columns:
    print("Error: 'text' column not found in the DataFrame. Please adjust the script to use the correct column name.")
    exit()

# Apply the entity extraction and company search
df = df.with_columns(
    pl.col('text').map_elements(extract_entities, return_dtype=pl.Object).alias('extracted_entities')
)

df = df.with_columns(
    pl.col('extracted_entities').map_elements(lambda x: x.get('PROFESSION', []), return_dtype=pl.List(pl.Utf8)).alias('PROFESSION'),
    pl.col('extracted_entities').map_elements(lambda x: x.get('FACILITY', []), return_dtype=pl.List(pl.Utf8)).alias('FACILITY'),
    pl.col('extracted_entities').map_elements(lambda x: x.get('EXPERIENCE', []), return_dtype=pl.List(pl.Utf8)).alias('EXPERIENCE'),
    pl.col('extracted_entities').map_elements(lambda x: x.get('MONEY', []), return_dtype=pl.List(pl.Utf8)).alias('MONEY'),
    pl.col('text').map_elements(company_searcher.find_company_names, return_dtype=pl.List(pl.Utf8)).alias('companies')
)

# Drop the intermediate 'extracted_entities' column if not needed
df = df.drop('extracted_entities')

# Save the labeled data to a new parquet file
output_file = 'candidate_searches_lpdoctor.parquet'
df.write_parquet(output_file)

print(f"Entities and companies extracted and saved to '{output_file}' with new columns: PROFESSION, FACILITY, EXPERIENCE, MONEY, COMPANIES.")