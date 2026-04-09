# PySpark Scalability Demonstration

import sys
import os

FIGURES_DIR = "figures"

def run_spark_pipeline(filepath):

    print("  PHASE 4: SPARK SCALABILITY DEMONSTRATION")

    # Patch JVM flags
    _java_opens = (
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
    )
    existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "--add-opens=java.base/javax.security.auth" not in existing:
        os.environ["JAVA_TOOL_OPTIONS"] = (existing + " " + _java_opens).strip()

    # Guard 1: pyspark import
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
        from pyspark.ml.classification import RandomForestClassifier as SparkRF
        from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                            MulticlassClassificationEvaluator)
        from pyspark.ml import Pipeline
        from pyspark.sql.functions import when, col
    except ImportError:
        print("\n  WARNING: pyspark is not installed.")
        print("  Install with: pip install pyspark")
        print("  Skipping Spark pipeline.")
        return None

    print("\n  Attempting to start Spark session...")

    # Guard 2: Java Runtime
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
        print(f"\n  WARNING: Could not start Spark session.")
        print(f"  Reason: {e}")
        print("\n  PySpark requires Java JDK/JRE 8 or 11+ to be installed.")
        print("  Skipping Spark pipeline - all other tasks completed successfully.")
        return None

    try:
        spark.sparkContext.setLogLevel("ERROR")
        print("  Spark session created successfully.")

        # Load Data
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        for c in df.columns:
            df = df.withColumnRenamed(c, c.strip())

        for drop_col in ['url', 'timedelta']:
            if drop_col in df.columns:
                df = df.drop(drop_col)

        print(f"  Loaded {df.count()} rows with {len(df.columns)} columns")

        # Binarize Target
        median_shares = df.approxQuantile('shares', [0.5], 0.001)[0]
        df = df.withColumn('label', when(col('shares') >= median_shares, 1.0).otherwise(0.0))
        df = df.drop('shares')
        print(f"  Binarized target: Popular if shares >= {median_shares:.0f}")

        # Build ML Pipeline
        feature_cols = [c for c in df.columns if c != 'label']
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_raw',
                                     handleInvalid='skip')
        scaler = SparkScaler(inputCol='features_raw', outputCol='features',
                              withStd=True, withMean=True)
        rf = SparkRF(featuresCol='features', labelCol='label',
                      numTrees=100, seed=42)

        pipeline = Pipeline(stages=[assembler, scaler, rf])

        # Train/Test Split
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        print(f"  Train: {train_df.count()} rows  |  Test: {test_df.count()} rows")

        # Train
        print("\n  Training Spark Random Forest...")
        model = pipeline.fit(train_df)
        print("  Training complete.")

        # Evaluate
        predictions = model.transform(test_df)
        binary_eval = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')
        multi_eval = MulticlassClassificationEvaluator(labelCol='label',
                                                        predictionCol='prediction',
                                                        metricName='accuracy')

        auc_val = binary_eval.evaluate(predictions)
        accuracy = multi_eval.evaluate(predictions)

        print(f"\n  Spark Random Forest Results")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  ROC AUC  : {auc_val:.4f}")
        print("\n  SPARK PIPELINE COMPLETE")


        return {'accuracy': accuracy, 'roc_auc': auc_val}

    except Exception as e:
        print(f"\n  WARNING: Spark pipeline encountered an error: {e}")
        print("  Skipping Spark pipeline.")
        return None

    finally:
        try:
            spark.stop()
            print("  Spark session stopped.")
        except Exception:
            pass
