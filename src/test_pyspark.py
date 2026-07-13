from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

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
    
    print("PySpark test completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
