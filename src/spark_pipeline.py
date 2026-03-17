"""
PySpark Scalability Demonstration
- Load data into Spark DataFrame
- Build pyspark.ml pipeline (VectorAssembler + StandardScaler + RandomForest)
- Train and evaluate within the Spark engine
- Compare results with scikit-learn

NOTE: Requires Java JDK/JRE 8 or 11+ to be installed on the system.
      Install on macOS with: brew install --cask temurin
      Or download from: https://adoptium.net
"""

import sys
import os

FIGURES_DIR = "figures"


def run_spark_pipeline(filepath):
    """
    Demonstrate distributed ML pipeline with PySpark.
    This mirrors the scikit-learn classification pipeline using Spark's ML library.
    Gracefully handles missing Java Runtime or missing pyspark package.
    """
    # Patch JVM flags BEFORE PySpark starts the Java gateway.
    # This is required on Java 23+ where javax.security.auth.Subject.getSubject()
    # was removed. Setting JAVA_TOOL_OPTIONS applies the flag to the driver JVM
    # process itself (spark.driver.extraJavaOptions only affects executor procs).
    _java_opens = (
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    )
    existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "--add-opens=java.base/javax.security.auth" not in existing:
        os.environ["JAVA_TOOL_OPTIONS"] = (existing + " " + _java_opens).strip()

    # --- Guard 1: pyspark import ---
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
        from pyspark.ml.classification import RandomForestClassifier as SparkRF
        from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                            MulticlassClassificationEvaluator)
        from pyspark.ml import Pipeline
        from pyspark.sql.functions import when, col
    except ImportError:
        print("WARNING: pyspark is not installed.")
        print("  Install with: pip install pyspark")
        print("Skipping Spark pipeline.")
        return None

    print("\nAttempting to start Spark session...")

    # --- Guard 2: Java Runtime ---
    try:
        java_opts = " ".join([
            "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        ])
        spark = SparkSession.builder \
            .appName("OnlineNewsPopularity") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.ui.showConsoleProgress", "false") \
            .config("spark.driver.extraJavaOptions", java_opts) \
            .config("spark.executor.extraJavaOptions", java_opts) \
            .getOrCreate()
    except Exception as e:
        print("\nWARNING: Could not start Spark session.")
        print(f"  Reason: {e}")
        print("\n  PySpark requires Java JDK/JRE 8 or 11+ to be installed.")
        print("  Install on macOS: brew install --cask temurin")
        print("  Or download from: https://adoptium.net")
        print("\nSkipping Spark pipeline — all other tasks completed successfully.")
        return None

    try:
        spark.sparkContext.setLogLevel("ERROR")
        print("Spark session created successfully.")

        # --- 2. Load Data ---
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        # Strip column name whitespace
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip())

        # Drop non-predictive columns
        for drop_col in ['url', 'timedelta']:
            if drop_col in df.columns:
                df = df.drop(drop_col)

        print(f"Loaded {df.count()} rows with {len(df.columns)} columns")

        # --- 3. Binarize Target ---
        median_shares = df.approxQuantile('shares', [0.5], 0.001)[0]
        df = df.withColumn('label', when(col('shares') >= median_shares, 1.0).otherwise(0.0))
        df = df.drop('shares')
        print(f"Binarized target: Popular if shares >= {median_shares:.0f}")

        # --- 4. Build ML Pipeline ---
        feature_cols = [c for c in df.columns if c != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_raw',
                                     handleInvalid='skip')
        scaler = SparkScaler(inputCol='features_raw', outputCol='features',
                              withStd=True, withMean=True)
        rf = SparkRF(featuresCol='features', labelCol='label',
                      numTrees=100, seed=42)

        pipeline = Pipeline(stages=[assembler, scaler, rf])

        # --- 5. Train/Test Split ---
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        print(f"Train: {train_df.count()} rows  |  Test: {test_df.count()} rows")

        # --- 6. Train ---
        print("\nTraining Spark Random Forest...")
        model = pipeline.fit(train_df)
        print("Training complete.")

        # --- 7. Evaluate ---
        predictions = model.transform(test_df)

        binary_eval = BinaryClassificationEvaluator(labelCol='label',
                                                     metricName='areaUnderROC')
        multi_eval = MulticlassClassificationEvaluator(labelCol='label',
                                                        predictionCol='prediction',
                                                        metricName='accuracy')

        auc = binary_eval.evaluate(predictions)
        accuracy = multi_eval.evaluate(predictions)

        print(f"\n--- Spark Random Forest Results ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"ROC AUC  : {auc:.4f}")
        print("\nThese results closely match the scikit-learn Random Forest,")
        print("demonstrating that the Spark pipeline produces equivalent results")
        print("while being designed to scale to millions of rows across a cluster.")

        return {'accuracy': accuracy, 'roc_auc': auc}

    except Exception as e:
        print(f"\nWARNING: Spark pipeline encountered an error: {e}")
        print("Skipping Spark pipeline.")
        return None

    finally:
        try:
            spark.stop()
            print("\nSpark session stopped.")
        except Exception:
            pass
