from pyspark.sql import SparkSession

class SparkSessionFactory:
    @staticmethod
    def create(app_name: str) -> SparkSession:
        spark = SparkSession.builder.appName(app_name).getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        return spark