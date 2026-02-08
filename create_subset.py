#!/usr/bin/env python3
"""
Script to create a deterministic 100-row subset from the UCI Online News Popularity dataset.
This script will attempt to download the dataset using ucimlrepo, and if that fails,
it will allow reading from a local CSV file.

Usage:
    python create_subset.py                    # Download from UCI and create subset
    python create_subset.py <local_csv_path>   # Create subset from local CSV
"""

import sys
import pandas as pd
import os

# Fixed random seed for reproducibility
RANDOM_SEED = 42
SUBSET_SIZE = 100
OUTPUT_FILE = "data/subset.csv"


def create_subset_from_uci():
    """
    Download the UCI Online News Popularity dataset and create a subset.
    
    Returns:
        pandas DataFrame with the subset, or None if failed
    """
    try:
        from ucimlrepo import fetch_ucirepo
        
        print("Fetching UCI Online News Popularity dataset (ID=332)...")
        news_popularity = fetch_ucirepo(id=332)
        
        # Get features and target
        X = news_popularity.data.features
        y = news_popularity.data.targets
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Create deterministic subset
        subset_df = df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
        
        return subset_df
    except ImportError:
        print("Error: ucimlrepo package not installed.")
        print("Install it with: pip install ucimlrepo")
        return None
    except Exception as e:
        print(f"Error fetching dataset from UCI: {e}")
        return None


def create_subset_from_csv(csv_path):
    """
    Load a local CSV file and create a subset.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the subset, or None if failed
    """
    try:
        print(f"Loading data from local file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Create deterministic subset
        subset_df = df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
        
        return subset_df
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def save_subset(subset_df, output_path):
    """
    Save the subset to a CSV file.
    
    Args:
        subset_df: pandas DataFrame to save
        output_path: Path where to save the file
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save to CSV
    subset_df.to_csv(output_path, index=False)
    print(f"Subset saved to: {output_path}")
    print(f"Subset size: {subset_df.shape[0]} rows, {subset_df.shape[1]} columns")


def main():
    """
    Main function to create the subset.
    """
    print("=" * 70)
    print("Creating deterministic 100-row subset from UCI Online News Popularity")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)
    print()
    
    # Check if a local CSV path was provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        subset_df = create_subset_from_csv(csv_path)
    else:
        subset_df = create_subset_from_uci()
    
    # If subset creation failed, exit
    if subset_df is None:
        print("\nFailed to create subset.")
        sys.exit(1)
    
    # Save the subset
    print()
    save_subset(subset_df, OUTPUT_FILE)
    print()
    print("Success! You can now run the analysis with:")
    print(f"  python report_1.py {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
