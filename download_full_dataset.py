#!/usr/bin/env python3
"""
Download the full UCI Online News Popularity dataset and save it as CSV.

Usage: python download_full_dataset.py
"""

import pandas as pd
import os

OUTPUT_FILE = "data/full_dataset.csv"


def download_full_dataset():
    """
    Download the full UCI Online News Popularity dataset.
    
    Returns:
        pandas DataFrame with the full dataset, or None if failed
    """
    try:
        from ucimlrepo import fetch_ucirepo
        
        print("Fetching full UCI Online News Popularity dataset (ID=332)...")
        print("This may take a few moments...")
        
        news_popularity = fetch_ucirepo(id=332)
        
        # Get features and target
        X = news_popularity.data.features
        y = news_popularity.data.targets
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    except ImportError:
        print("Error: ucimlrepo package not installed.")
        print("Install it with: pip install ucimlrepo")
        return None
    except Exception as e:
        print(f"Error fetching dataset from UCI: {e}")
        return None


def save_full_dataset(df, output_path):
    """
    Save the full dataset to a CSV file.
    
    Args:
        df: pandas DataFrame to save
        output_path: Path where to save the file
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save to CSV
    print(f"Saving full dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Full dataset saved to: {output_path}")
    print(f"   Size: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


def main():
    """
    Main function to download and save the full dataset.
    """
    print("=" * 70)
    print("Downloading Full UCI Online News Popularity Dataset")
    print("=" * 70)
    print()
    
    # Download full dataset
    df = download_full_dataset()
    
    # If download failed, exit
    if df is None:
        print("\nFailed to download full dataset.")
        return
    
    # Save the full dataset
    print()
    save_full_dataset(df, OUTPUT_FILE)
    print()
    print(" Success! You can now run the analysis on the full dataset with:")
    print(f"  python report_1.py {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
