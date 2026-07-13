import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from huggingface_hub import HfApi

def main():
    print("Starting PySpark session...")
    spark = SparkSession.builder \
        .appName("GitHubActionPySparkTest") \
        .master("local[*]") \
        .getOrCreate()
        
    print("Spark Session created successfully.")
    
    data = [("Alice", 34), ("Bob", 45), ("Charlie", 28)]
    schema = StructType([
        StructField("Name", StringType(), True),
        StructField("Age", IntegerType(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    print("DataFrame created:")
    df.show()
    
    # Save the DataFrame to a local JSON file
    output_path = "output_data"
    print(f"Saving DataFrame to {output_path} as JSON...")
    # coalesce(1) ensures it writes out to a single file instead of partitioning
    df.coalesce(1).write.mode("overwrite").json(output_path)
    
    # PySpark writes inside a directory, let's find the actual .json file
    json_files = [f for f in os.listdir(output_path) if f.endswith(".json")]
    if not json_files:
        raise Exception("No JSON file found in output directory.")
    
    local_file_path = os.path.join(output_path, json_files[0])
    print(f"Generated JSON file at: {local_file_path}")
    
    # Upload to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if hf_token and hf_repo_id:
        print(f"Uploading {local_file_path} to Hugging Face repo: {hf_repo_id}...")
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo="pyspark_output.json",
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        print("Upload to Hugging Face completed successfully!")
    else:
        print("Skipping Hugging Face upload: HF_TOKEN or HF_REPO_ID environment variables are not set.")
    
    print("PySpark test completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
