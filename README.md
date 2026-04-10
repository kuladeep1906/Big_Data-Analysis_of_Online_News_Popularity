# Big Data Course Project

## UCI Online News Popularity Dataset Analysis

```bash
# 1. Install Java 17 LTS (required for PySpark)
brew install --cask temurin@17
export JAVA_HOME=$(/usr/libexec/java_home -v 17) 
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc  

# 2. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) Download data from UCI if you do not have a CSV yet
python download_full_dataset.py

# 4. Run the 500-row subset
python report_2_task.py data/subset.csv

# 5. Run the full analysis pipeline
python report_2_task.py data/full_dataset.csv
```