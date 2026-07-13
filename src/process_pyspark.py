import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from huggingface_hub import HfFileSystem

def main():
    print("Starting PySpark Transformation Session...")
    # Initialize PySpark
    spark = SparkSession.builder \
        .appName("GitHubActionPySparkProcess") \
        .master("local[*]") \
        .getOrCreate()
        
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID") # e.g., aayushbhat26/testBucket
    
    if not hf_token or not hf_repo_id:
        raise Exception("HF_TOKEN or HF_REPO_ID environment variables are missing!")

    fs = HfFileSystem(token=hf_token)
    
    # ---------------------------------------------------------
    # 1. FETCH DATA FROM HUGGING FACE BUCKET
    # ---------------------------------------------------------
    source_bucket_path = f"hf://buckets/{hf_repo_id}/pyspark_output.json"
    local_input_path = "downloaded_data.json"
    
    print(f"Downloading data from {source_bucket_path}...")
    fs.get(source_bucket_path, local_input_path)
    print("Download complete.")
    
    # ---------------------------------------------------------
    # 2. READ & TRANSFORM DATA WITH PYSPARK
    # ---------------------------------------------------------
    print("Loading data into PySpark DataFrame...")
    df = spark.read.json(local_input_path)
    
    print("Original Data (Fetched from Hugging Face):")
    df.show()
    
    print("Applying Transformations...")
    # Example Transformation: Add a column and filter out rows
    transformed_df = df.withColumn("Age_in_10_Years", col("Age") + 10) \
                       .withColumn("Status", lit("Processed by GitHub Actions")) \
                       .filter(col("Age") > 30)
                       
    print("Transformed Data:")
    transformed_df.show()
    
    # ---------------------------------------------------------
    # 3. SAVE TRANSFORMED DATA LOCALLY
    # ---------------------------------------------------------
    output_dir = "transformed_data"
    print(f"Saving transformed data to {output_dir}...")
    transformed_df.coalesce(1).write.mode("overwrite").json(output_dir)
    
    # Find the newly generated JSON file
    json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
    local_output_path = os.path.join(output_dir, json_files[0])
    
    # ---------------------------------------------------------
    # 4. UPLOAD BACK TO HUGGING FACE BUCKET
    # ---------------------------------------------------------
    destination_bucket_path = f"hf://buckets/{hf_repo_id}/transformed_output.json"
    print(f"Uploading transformed data back to {destination_bucket_path}...")
    
    fs.put(local_output_path, destination_bucket_path)
    
    print("Process complete! Transformed data successfully stored back into the Hugging Face Bucket.")
    spark.stop()

if __name__ == "__main__":
    main()
