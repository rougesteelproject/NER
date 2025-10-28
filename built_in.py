import polars as pl
import spacy
import re
import json
import os
import pickle

# Load SpaCy's default English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_gpe_spacy(text):
    """
    Processes text with SpaCy and extracts GPE (Geo-Political Entity) labels.
    Returns a comma-separated string of unique GPEs found.
    """
    if text is None:
        return None
    doc = nlp(str(text))
    gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return ", ".join(sorted(list(set(gpes))))

def extract_distance_regex(text):
    """
    Uses regex to extract distance information from text.
    Returns a comma-separated string of unique distance phrases found.
    """
    if text is None:
        return None
    # Regex pattern to find phrases like "within 10 miles", "up to 5 km"
    pattern = r"\b(within|up to)\s+(\d+)\s+(miles|km|kilometers)\b"
    matches = re.findall(pattern, str(text), re.IGNORECASE)
    # Reconstruct the matched phrases from the captured groups
    distances = [f"{match[0]} {match[1]} {match[2]}" for match in matches]
    return ", ".join(sorted(list(set(distances))))

def extract_annual_salary_regex(text):
    """
    Uses regex to extract salary information and normalizes it to an annual figure.
    Returns the annual salary as a float or None if not found.
    """
    if text is None:
        return None

    text_lower = str(text).lower()

    # Pre-filter to exclude phrases like "X years experience"
    if re.search(r"\d+\s*(?:year|yrs?)\s+experience", text_lower):
        return None

    annual_salaries = []

    # Refined regex pattern for salary, requiring currency or explicit salary terms,
    # and using a negative lookahead to avoid "experience" after "year".
    pattern_salary = r"(?:[\$€£]\s*|salary\s*of\s*|pay\s*of\s*|wage\s*of\s*)?(\d[\d,]*\.?\d*)\s*(k|thousand)?\s*(?:/|\b(per)\b)?\s*(hour|hr|day|week|month|year)?(?!\s*experience)"
    
    matches = re.findall(pattern_salary, text_lower)

    for match in matches:
        value_str = match[0].replace(",", "")
        try:
            value = float(value_str)
        except ValueError:
            continue

        multiplier = 1
        if match[1] in ("k", "thousand"):
            multiplier = 1000

        frequency = match[3] # hour, day, week, month, year

        annual_value = None
        if frequency in ("hour", "hr"):
            annual_value = value * multiplier * 40 * 52  # 40 hours/week, 52 weeks/year
        elif frequency == "day":
            annual_value = value * multiplier * 5 * 52   # 5 days/week, 52 weeks/year
        elif frequency == "week":
            annual_value = value * multiplier * 52
        elif frequency == "month":
            annual_value = value * multiplier * 12
        elif frequency == "year":
            annual_value = value * multiplier
        elif not frequency and (match[1] or value > 1000): # Assume annual if no frequency and large number or 'k'
            annual_value = value * multiplier
        
        if annual_value is not None:
            annual_salaries.append(annual_value)
    
    if annual_salaries:
        # Return the highest annual salary found
        return max(annual_salaries)
    return None


def extract_work_type_regex(text):
    """
    Uses regex to extract work type information from text.
    Returns a comma-separated string of unique work types found.
    """
    if text is None:
        return None
    text_lower = str(text).lower()
    # Regex pattern for common work types
    pattern = r"\b(remote|hybrid|on-site|in-office|telecommute)\b"
    matches = re.findall(pattern, text_lower)
    return ", ".join(sorted(list(set(matches))))

def extract_job_type_regex(text):
    """
    Uses regex to extract job type information from text.
    Returns a comma-separated string of unique job types found.
    """
    if text is None:
        return None
    text_lower = str(text).lower()
    # Regex pattern for common job types
    pattern = r"\b(full-time|part-time|contract|internship|freelance)\b"
    matches = re.findall(pattern, text_lower)
    return ", ".join(sorted(list(set(matches))))


def extract_company_spacy(text):
    """
    Processes text with SpaCy and extracts ORG (Organization) labels.
    Returns a comma-separated string of unique ORGs found.
    """
    if text is None:
        return None
    doc = nlp(str(text))
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return ", ".join(sorted(list(set(orgs))))


def extract_job_titles(text, job_titles_set):
    """
    Extracts job titles from text that match any title in the job_titles_set.
    Returns a comma-separated string of unique job titles found.
    """
    if text is None:
        return None

    # Find all job titles in the text
    found_titles = [title for title in job_titles_set if title.lower() in str(text).lower()]
    return ", ".join(sorted(list(set(found_titles))))

def load_job_titles():
    """
    Loads job titles from either job-titles.json or job-titles.txt.
    Returns a set of job titles.
    """
    try:
        # Try to load from JSON first
        with open("job-titles-master/job-titles.json", "r") as f:
            data = json.load(f)
            return set(data.get("job-titles", []))
    except:
        # Fall back to TXT file
        try:
            with open("job-titles-master/job-titles.txt", "r") as f:
                return set(line.strip() for line in f if line.strip())
        except:
            return set()


def main():
    # Read the Parquet file
    df = pl.read_parquet("candidate_searches_copy.parquet")

    # Apply the SpaCy GPE extraction function
    df = df.with_columns(
        pl.col("text").map_elements(extract_gpe_spacy, return_dtype=pl.String).alias("LOCATION_SpaCy")
    )

    # Apply the regex distance extraction function
    df = df.with_columns(
        pl.col("text").map_elements(extract_distance_regex, return_dtype=pl.String).alias("DISTANCE_RegEx")
    )

    # Apply the regex annual salary extraction function
    df = df.with_columns(
        pl.col("text").map_elements(extract_annual_salary_regex, return_dtype=pl.Float64).alias("ANNUAL_SALARY_RegEx")
    )

    # Apply the regex work type extraction function
    df = df.with_columns(
        pl.col("text").map_elements(extract_work_type_regex, return_dtype=pl.String).alias("WORK_TYPE_RegEx")
    )

    # Apply the regex job type extraction function
    df = df.with_columns(
        pl.col("text").map_elements(extract_job_type_regex, return_dtype=pl.String).alias("JOB_TYPE_RegEx")
    )

    # Apply the SpaCy company extraction function
    df = df.with_columns(
        pl.col("text").map_elements(extract_company_spacy, return_dtype=pl.String).alias("COMPANY_SpaCy")
    )

    job_titles_set = load_job_titles()

    # Apply the job title extraction function
    df = df.with_columns(
        pl.col("text").map_elements(lambda text: extract_job_titles(text, job_titles_set), return_dtype=pl.String).alias("TITLE_dataset")
    )

    # Save the DataFrame to a new Parquet file
    df.write_parquet("candidate_searches_built_in.parquet")
    print("DataFrame with LOCATION_SpaCy, DISTANCE_RegEx, ANNUAL_SALARY_RegEx, WORK_TYPE_RegEx, JOB_TYPE_RegEx, COMPANY_SpaCy, TITLE_dataset, columns saved to candidate_searches_built_in.parquet")

if __name__ == "__main__":
    main()