import pandas as pd
import re
import csv
from io import StringIO

def clean_csv_file(input_file: str, output_file: str = None):
    """
    Clean messy CSV file with excessive quotes and formatting issues
    
    Args:
        input_file: Path to the messy CSV file
        output_file: Path for cleaned output (optional, defaults to input_file_cleaned.csv)
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_cleaned.csv')
    
    print(f"Cleaning CSV file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    try:
        # Read the raw file content
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()
        
        # Clean up the raw content
        print("Step 1: Cleaning raw content...")
        
        # Remove excessive quotes (3 or more consecutive quotes)
        cleaned_content = re.sub(r'"{3,}', '"', raw_content)
        
        # Fix lines that start with quotes after comma
        cleaned_content = re.sub(r',\s*"([^"]*)"', r',\1', cleaned_content)
        
        # Remove quotes around numbers
        cleaned_content = re.sub(r'"(\d+)"', r'\1', cleaned_content)
        
        # Clean up null values
        cleaned_content = re.sub(r'"null"', 'null', cleaned_content)
        
        # Fix malformed lines (lines that don't end properly)
        lines = cleaned_content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                # Count quotes in line
                quote_count = line.count('"')
                
                # If odd number of quotes, there's likely a formatting issue
                if quote_count % 2 != 0 and i > 0:  # Skip header
                    print(f"Warning: Line {i+1} has uneven quotes, attempting to fix...")
                    # Try to fix by removing trailing quote or adding one
                    if line.endswith('"'):
                        line = line[:-1]
                    else:
                        line = line + '"'
                
                fixed_lines.append(line)
        
        cleaned_content = '\n'.join(fixed_lines)
        
        # Try to parse with pandas
        print("Step 2: Parsing with pandas...")
        
        try:
            # Use StringIO to read the cleaned content
            df = pd.read_csv(
                StringIO(cleaned_content),
                quotechar='"',
                skipinitialspace=True,
                on_bad_lines='skip',
                dtype=str  # Read everything as string initially
            )
            
            print(f"Successfully parsed {len(df)} rows")
            print(f"Columns found: {list(df.columns)}")
            
        except Exception as e:
            print(f"Pandas parsing failed: {e}")
            print("Attempting manual CSV parsing...")
            
            # Manual parsing as fallback
            csv_reader = csv.reader(StringIO(cleaned_content), quotechar='"')
            rows = []
            headers = None
            
            for i, row in enumerate(csv_reader):
                if i == 0:
                    headers = [col.strip().strip('"') for col in row]
                else:
                    # Clean each cell
                    cleaned_row = [cell.strip().strip('"') if cell else '' for cell in row]
                    rows.append(cleaned_row)
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"Manual parsing successful: {len(df)} rows")
        
        # Clean up the dataframe
        print("Step 3: Cleaning dataframe...")
        
        # Clean column names
        df.columns = [col.strip().strip('"') for col in df.columns]
        
        # Clean all string data
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
                # Remove multiple quotes
                df[col] = df[col].str.replace(r'"{2,}', '"', regex=True)
                # Remove leading/trailing quotes and spaces
                df[col] = df[col].str.strip().str.strip('"')
                # Handle null values
                df[col] = df[col].replace(['null', 'NULL', 'None'], '')
        
        # Convert numeric columns based on known schema
        numeric_columns = ['n.id', 'n.birthyear', 'n.deathyear', 'n.numCities', 'n.numCountries', 'n.numVenues']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create clean artist_name column from n.firstname and n.lastname
        if 'n.firstname' in df.columns and 'n.lastname' in df.columns:
            df['artist_name'] = (
                df['n.firstname'].fillna('').astype(str) + ' ' + 
                df['n.lastname'].fillna('').astype(str)
            ).str.strip()
            
            # Remove double spaces and clean up
            df['artist_name'] = df['artist_name'].str.replace(r'\s+', ' ', regex=True)
            # Remove entries that are just spaces
            df['artist_name'] = df['artist_name'].replace('', None)
        
        # Clean up wikidata column for easier use
        if 'n.wikidata' in df.columns:
            df['wiki_id'] = df['n.wikidata'].str.replace('Q', '', regex=False)
            # Remove empty wiki_ids
            df['wiki_id'] = df['wiki_id'].replace('', None)
        
        # Clean up other important text columns
        text_columns = ['n.birthplace', 'n.deathplace', 'n.country', 'n.collective', 'n.ulan']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].replace('', None)
        
        # Save cleaned CSV
        print("Step 4: Saving cleaned CSV...")
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
        
        print(f"✅ Successfully cleaned CSV!")
        print(f"Original rows: {len(fixed_lines)-1}")  # -1 for header
        print(f"Cleaned rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample of cleaned data
        print("\nSample cleaned data:")
        sample_cols = ['artist_name', 'n.birthyear', 'n.deathyear', 'n.country', 'n.wikidata', 'wiki_id']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head())
        
        return output_file
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
        return None
    except Exception as e:
        print(f"❌ Error cleaning CSV: {e}")
        return None

def inspect_csv_issues(input_file: str):
    """
    Inspect the CSV file to identify formatting issues
    
    Args:
        input_file: Path to the CSV file to inspect
    """
    print(f"Inspecting CSV file: {input_file}")
    print("=" * 50)
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        # Check first few lines
        print("\nFirst 3 lines:")
        for i, line in enumerate(lines[:3]):
            print(f"Line {i+1}: {repr(line[:100])}")  # Show first 100 chars
        
        # Check for quote issues
        quote_issues = []
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            quote_count = line.count('"')
            if quote_count > 20:  # Excessive quotes
                quote_issues.append((i+1, quote_count))
        
        if quote_issues:
            print(f"\nLines with excessive quotes (>20):")
            for line_num, count in quote_issues:
                print(f"  Line {line_num}: {count} quotes")
        
        # Check line lengths
        line_lengths = [len(line) for line in lines[:100]]  # First 100 lines
        avg_length = sum(line_lengths) / len(line_lengths)
        max_length = max(line_lengths)
        
        print(f"\nLine length stats (first 100 lines):")
        print(f"  Average: {avg_length:.1f} characters")
        print(f"  Maximum: {max_length} characters")
        
        # Try to detect delimiter
        first_data_line = lines[1] if len(lines) > 1 else ""
        comma_count = first_data_line.count(',')
        semicolon_count = first_data_line.count(';')
        
        print(f"\nDelimiter detection:")
        print(f"  Commas in line 2: {comma_count}")
        print(f"  Semicolons in line 2: {semicolon_count}")
        print(f"  Expected columns: 15 (based on known schema)")
        
        # Check if we have the expected columns
        if len(lines) > 0:
            header_line = lines[0]
            expected_cols = ['n.id', 'n.firstname', 'n.lastname', 'n.birthyear', 'n.deathyear', 
                           'n.birthplace', 'n.deathplace', 'n.sex', 'n.numCities', 'n.numCountries', 
                           'n.numVenues', 'n.country', 'n.collective', 'n.ulan', 'n.wikidata']
            
            print(f"\nExpected header columns:")
            for col in expected_cols:
                if col in header_line:
                    print(f"  ✅ {col} - found")
                else:
                    print(f"  ❌ {col} - missing")
        
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = "ArtRag_ESSIR/artists1000.csv"
    
    # First, inspect the file to see what issues we're dealing with
    inspect_csv_issues(input_file)
    
    print("\n" + "=" * 50)
    
    # Clean the file
    cleaned_file = clean_csv_file(input_file)
    
    if cleaned_file:
        print(f"\n✅ Cleaned file saved as: {cleaned_file}")
        print("You can now use this cleaned file with the main RAG pipeline!")
    else:
        print("\n❌ Failed to clean the file")