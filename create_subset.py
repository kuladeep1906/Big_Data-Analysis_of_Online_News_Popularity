#!/usr/bin/env python3

import sys
import pandas as pd
import os

RANDOM_SEED = 42
SUBSET_SIZE = 500
OUTPUT_FILE = "data/subset.csv"


def create_subset_from_uci():
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

    try:
        print(f"Loading data from local file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
   
        subset_df = df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
        
        return subset_df
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def save_subset(subset_df, output_path):
   # Save the subset to a CSV file.
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    subset_df.to_csv(output_path, index=False)
    print(f"Subset saved to: {output_path}")
    print(f"Subset size: {subset_df.shape[0]} rows, {subset_df.shape[1]} columns")


def main():
   
 
    print("Creating subset ")
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
    
    # Save subset
    print()
    save_subset(subset_df, OUTPUT_FILE)
    print()
    print("Success! You can now run the analysis")


if __name__ == "__main__":
    main()
