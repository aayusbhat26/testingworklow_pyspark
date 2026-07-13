import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from huggingface_hub import HfFileSystem, sync_bucket
from delta import *

def main():
    print("Starting PySpark Medallion Transformation Session (Delta Lake)...")
    
    # Initialize PySpark with Delta Lake configurations
    builder = SparkSession.builder \
        .appName("GitHubActionPySparkProcess") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    # configure_spark_with_delta_pip helps to inject delta-spark dependencies
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
        
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID") # e.g., aayushbhat26/testBucket
    
    if not hf_token or not hf_repo_id:
        raise Exception("HF_TOKEN or HF_REPO_ID environment variables are missing!")

    fs = HfFileSystem(token=hf_token)
    
    # ---------------------------------------------------------
    # 1. READ RAW DATA (JSON)
    # ---------------------------------------------------------
    source_bucket_path = f"hf://buckets/{hf_repo_id}/pyspark_output.json"
    local_input_path = "raw_data.json"
    
    print(f"Downloading raw JSON data from {source_bucket_path}...")
    try:
        fs.get(source_bucket_path, local_input_path)
    except FileNotFoundError:
        print("Warning: pyspark_output.json not found, using dummy data instead.")
        with open(local_input_path, 'w') as f:
            f.write('{"Name":"Alice","Age":34}\n{"Name":"Bob","Age":45}\n{"Name":"Charlie","Age":28}\n')

    raw_df = spark.read.json(local_input_path)
    print("Raw Data (Bronze):")
    raw_df.show()
    
    # ---------------------------------------------------------
    # 2. CREATE SILVER TABLE (DELTA)
    # ---------------------------------------------------------
    print("\n--- Transforming Raw -> Silver ---")
    silver_df = raw_df.withColumn("Status", lit("Processed into Silver"))
    
    print("Silver Table Schema:")
    silver_df.printSchema()
    print("Silver Table Data:")
    silver_df.show()
    
    local_silver_dir = "silver_table.delta"
    print(f"Writing Silver table locally to {local_silver_dir}...")
    silver_df.write.format("delta").mode("overwrite").save(local_silver_dir)
    
    # Print Delta Table History
    print("Silver Delta Table History:")
    DeltaTable.forPath(spark, local_silver_dir).history().select("version", "timestamp", "operation", "operationParameters").show(truncate=False)
    
    silver_dest = f"hf://buckets/{hf_repo_id}/silver_table.delta"
    print(f"Uploading Silver table to {silver_dest} using sync_bucket...")
    sync_bucket(local_silver_dir, silver_dest, token=hf_token)
    
    # ---------------------------------------------------------
    # 3. CREATE GOLD TABLE (DELTA)
    # ---------------------------------------------------------
    print("\n--- Transforming Silver -> Gold ---")
    gold_df = silver_df.filter(col("Age") > 30).withColumn("Age_in_10_Years", col("Age") + 10)
    
    print("Gold Table Schema:")
    gold_df.printSchema()
    print("Gold Table Data:")
    gold_df.show()
    
    local_gold_dir = "gold_table.delta"
    print(f"Writing Gold table locally to {local_gold_dir}...")
    gold_df.write.format("delta").mode("overwrite").save(local_gold_dir)
    
    # Print Delta Table History
    print("Gold Delta Table History:")
    DeltaTable.forPath(spark, local_gold_dir).history().select("version", "timestamp", "operation", "operationParameters").show(truncate=False)
    
    gold_dest = f"hf://buckets/{hf_repo_id}/gold_table.delta"
    print(f"Uploading Gold table to {gold_dest} using sync_bucket...")
    sync_bucket(local_gold_dir, gold_dest, token=hf_token)
    
    print("\nMedallion pipeline complete! Silver and Gold Delta tables stored in Hugging Face Bucket.")
    spark.stop()

if __name__ == "__main__":
    main()
