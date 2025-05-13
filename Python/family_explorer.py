import pandas as pd
import os

def extract_families(file_path):
    """
    Extract unique family names from a GBIF CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        list: Sorted list of unique family names
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path, sep='\t')
        
        # Extract unique families, remove NaN values and sort
        families = sorted(df['family'].dropna().unique())
        
        return families
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def main():
    # Directory containing GBIF data files
    data_dir = os.path.join(os.path.dirname(__file__), "MadHornet")
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_families = set()
    
    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        print(f"\nProcessing: {csv_file}")
        
        families = extract_families(file_path)
        all_families.update(families)
        
        print(f"Found {len(families)} families in this file")
    
    # Sort the combined families
    all_families = sorted(all_families)
    
    # Print results
    print(f"\nTotal unique families found: {len(all_families)}")
    print("\nList of all families:")
    for family in all_families:
        print(family)
    
    # Write results to file - changed to save in MadHornet directory
    output_file = os.path.join(data_dir, "gbif_families.txt")
    with open(output_file, 'w') as f:
        f.write(f"Total families: {len(all_families)}\n\n")
        f.write("\n".join(all_families))
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main()
