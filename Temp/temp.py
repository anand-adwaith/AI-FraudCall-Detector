import pandas as pd
import os
from pathlib import Path

def merge_text_csvs():
    """
    Read text_1.csv, text_2.csv, text_3.csv from the Dataset folder,
    merge them, remove duplicate rows, and reorder columns as 
    Label, Type, Text.
    
    Returns:
        pd.DataFrame: The merged and processed DataFrame
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "Dataset"
    
    # Define the input files
    input_files = [
        dataset_dir / "text_1.csv",
        dataset_dir / "text_2.csv",
        dataset_dir / "text_3.csv"
    ]
    
    # Read and concatenate all CSV files
    dfs = []
    for file in input_files:
        if file.exists():
            print(f"Reading file: {file}")
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            print(f"Warning: File not found - {file}")
    
    if not dfs:
        raise FileNotFoundError("No input CSV files were found")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Check original size
    original_size = len(merged_df)
    print(f"Original number of rows: {original_size}")
    
    # Remove duplicates based on all columns
    merged_df.drop_duplicates(inplace=True)
    
    # Check size after removing duplicates
    new_size = len(merged_df)
    print(f"Number of rows after removing duplicates: {new_size}")
    print(f"Removed {original_size - new_size} duplicate rows")
    
    # Rename columns
    column_mapping = {
        "Type": "Label",
        "Category": "Type",
        "Message": "Text"
    }
    
    merged_df.rename(columns=column_mapping, inplace=True)
    
    # Reorder columns
    column_order = ["Label", "Type", "Text"]
    merged_df = merged_df[column_order]
    
    # Display sample of the processed data
    print("\nSample of processed data:")
    print(merged_df.head())
    
    # Save the merged data to a new CSV file
    output_path = dataset_dir / "merged_text_data.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
    
    return merged_df

def merge_call_data_csvs():
    """
    Read data_call.csv and data.csv from the Dataset folder,
    standardize column names, merge them, remove duplicate rows,
    and reorder columns as Label, Type, Text.
    
    Returns:
        pd.DataFrame: The merged and processed DataFrame
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "Dataset"
    
    # Define the input files
    data_call_path = dataset_dir / "data_call.csv"
    data_path = dataset_dir / "data.csv"
    
    # Read data_call.csv
    if data_call_path.exists():
        print(f"Reading file: {data_call_path}")
        data_call_df = pd.read_csv(data_call_path)
        # Rename columns for data_call.csv: Call_Transcript -> Text, Label -> Label, Category -> Type
        data_call_df.rename(columns={
            "Call_Transcript": "Text",
            "Category": "Type"
            # "Label" stays as "Label"
        }, inplace=True)
    else:
        print(f"Warning: File not found - {data_call_path}")
        data_call_df = None
    
    # Read data.csv
    if data_path.exists():
        print(f"Reading file: {data_path}")
        data_df = pd.read_csv(data_path)
        # Rename columns for data.csv: text -> Text, label -> Label, scam_category -> Type
        data_df.rename(columns={
            "text": "Text",
            "label": "Label",
            "scam_category": "Type"
        }, inplace=True)
    else:
        print(f"Warning: File not found - {data_path}")
        data_df = None
    
    # Check if we have at least one dataframe
    if data_call_df is None and data_df is None:
        raise FileNotFoundError("No input CSV files were found")
    
    # Merge dataframes if both exist
    if data_call_df is not None and data_df is not None:
        # Concatenate all dataframes
        merged_df = pd.concat([data_call_df, data_df], ignore_index=True)
    elif data_call_df is not None:
        merged_df = data_call_df
    else:
        merged_df = data_df
    
    # Check original size
    original_size = len(merged_df)
    print(f"Original number of rows: {original_size}")
    
    # Remove duplicates based on all columns
    merged_df.drop_duplicates(inplace=True)
    
    # Check size after removing duplicates
    new_size = len(merged_df)
    print(f"Number of rows after removing duplicates: {new_size}")
    print(f"Removed {original_size - new_size} duplicate rows")
    
    # Reorder columns
    column_order = ["Label", "Type", "Text"]
    merged_df = merged_df[column_order]
    
    # Display sample of the processed data
    print("\nSample of processed data:")
    print(merged_df.head())
    
    # Save the merged data to a new CSV file
    output_path = dataset_dir / "merged_call_data.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")
    
    return merged_df

if __name__ == "__main__":
    print("Starting CSV merge process...")
    try:
        # print("\n=== Merging text_1.csv, text_2.csv, text_3.csv ===")
        # merged_text_data = merge_text_csvs()
        
        print("\n=== Merging data_call.csv and data.csv ===")
        merged_call_data = merge_call_data_csvs()
        
        print("\nBoth processes completed successfully!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
