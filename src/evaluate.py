import json
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_session import SparkSessionFactory

from logger import Logger

class ClusterEvaluator:
    def __init__(self, spark: SparkSession, config_path: str="config.json") -> None:
        self.spark = spark
        self.config = self._load_config(config_path)

        logger = Logger(show=True)
        self.log = logger.get_logger(__name__)

        model_config = self.config["model"]
        self.feature_names = self.config["model_features"]
        self.predictions_path = model_config["predictions_path"]
        self.profiles_path = model_config["profiles_path"]
        self.categories_path = model_config["categories_path"]
        self.examples_path = model_config["examples_path"]
        self.seed = model_config["seed"]

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config was not found: {path}")

        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def read_predictions(self) -> DataFrame:
        self.log.info("Reading predictions: %s", self.predictions_path)

        predictions = self.spark.read.parquet(self.predictions_path)
        self.log.info("Prediction partitions: %d", predictions.rdd.getNumPartitions())

        return predictions

    def create_cluster_profiles(self, predictions: DataFrame) -> DataFrame:
        return (
            predictions.groupBy("prediction")
            .agg(F.count("*").alias("products_count"),
                 *[F.round(F.avg(feature), 3).alias(f"avg_{feature.replace('-', '_')}")
                   for feature in self.feature_names])
            .orderBy("prediction")
        )

    @staticmethod
    def create_category_profiles(predictions: DataFrame) -> DataFrame:
        category_rank = (
            Window
            .partitionBy("prediction")
            .orderBy(F.desc("products_count"), F.asc("category"))
        )

        return (
            predictions
            .where(F.col("category").isNotNull() & (F.trim(F.col("category")) != ""))
            .groupBy("prediction", "category")
            .agg(F.count("*").alias("products_count"))
            .withColumn("rank", F.row_number().over(category_rank))
            .where(F.col("rank") <= 5)
            .select(
                "prediction",
                "rank",
                "category",
                "products_count",
            )
            .orderBy("prediction", "rank")
        )

    def create_product_example(self, predictions: DataFrame) -> DataFrame:
        example_rank = Window.partitionBy("prediction").orderBy("random_order")

        return (
            predictions
            .where(F.col("product_name").isNotNull() & (F.trim(F.col("product_name")) != ""))
            .withColumn("random_order", F.rand(self.seed))
            .withColumn("rank", F.row_number().over(example_rank))
            .where(F.col("rank") <= 10)
            .select(
                "prediction",
                "rank",
                "product_name",
                "brands",
                "category",
                "food_group",
                *self.feature_names,
            )
            .orderBy("prediction", "rank")
        )

    def save_results(self, profiles: DataFrame, categories: DataFrame, examples: DataFrame) -> None:
        profiles.coalesce(1).write.mode("overwrite").option("header", True).csv(self.profiles_path)
        categories.coalesce(1).write.mode("overwrite").option("header", True).csv(self.categories_path)
        examples.coalesce(1).write.mode("overwrite").option("header", True).csv(self.examples_path)

        self.log.info("Cluster profiles saved: %s", self.profiles_path)
        self.log.info("Cluster categories saved: %s", self.categories_path)
        self.log.info("Cluster examples saved: %s", self.examples_path)

    def run(self) -> None:
        predictions = self.read_predictions()
        predictions.persist(StorageLevel.MEMORY_AND_DISK)

        try:
            predictions_count = predictions.count()
            cluster_count = predictions.select("prediction").distinct().count()

            self.log.info("Prediction rows: %d", predictions_count)
            self.log.info("Clusters found: %d", cluster_count)

            profiles = self.create_cluster_profiles(predictions)
            categories = self.create_category_profiles(predictions)
            examples = self.create_product_example(predictions)

            print("\nCluster profiles:")
            profiles.show(cluster_count, truncate=False)

            print("\nTop categories by cluster:")
            categories.show(cluster_count * 5, truncate=False)

            print("\nProduct examples by cluster:")
            examples.show(cluster_count * 10, truncate=False, vertical=True)

            self.save_results(profiles, categories, examples)
        finally:
            predictions.unpersist()


def main() -> None:
    spark = SparkSessionFactory.create(app_name="OpenFoodFactsEvaluation")

    try:
        evaluator = ClusterEvaluator(spark=spark)
        evaluator.run()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()