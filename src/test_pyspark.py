import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pyxet
import fsspec

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
    df.coalesce(1).write.mode("overwrite").json(output_path)
    
    json_files = [f for f in os.listdir(output_path) if f.endswith(".json")]
    if not json_files:
        raise Exception("No JSON file found in output directory.")
    
    local_file_path = os.path.join(output_path, json_files[0])
    print(f"Generated JSON file at: {local_file_path}")
    
    # Upload to XetHub (Bucket)
    xet_user = os.environ.get("XET_USER_NAME")
    xet_token = os.environ.get("XET_USER_TOKEN")
    xet_repo = os.environ.get("XET_REPO_ID")
    
    if xet_user and xet_token and xet_repo:
        print(f"Uploading {local_file_path} to XetHub bucket: {xet_repo}...")
        
        # Authenticate with XetHub
        pyxet.login(user=xet_user, token=xet_token)
        
        # XetHub path format: xet://username/repo/branch/filename
        destination = f"xet://{xet_repo}/main/pyspark_output.json"
        
        # Upload using fsspec
        with fsspec.open(destination, 'wb') as f_dest:
            with open(local_file_path, 'rb') as f_source:
                f_dest.write(f_source.read())
                
        print("Upload to XetHub completed successfully!")
    else:
        print("Skipping XetHub upload: XET_USER_NAME, XET_USER_TOKEN, or XET_REPO_ID environment variables are not set.")
    
    print("PySpark test completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
