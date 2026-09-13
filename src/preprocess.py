import json
from functools import reduce
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_session import SparkSessionFactory

from logger import Logger


class OpenFoodFactsPreprocessor:
    def __init__(self, spark: SparkSession, config_path: str = "config.json") -> None:
        self.spark = spark
        self.config = self._load_config(config_path)

        logger = Logger(show=True)
        self.log = logger.get_logger(__name__)

        self.input_path = self.config["data"]["input_path"]
        self.output_path = self.config["data"]["output_path"]
        self.max_rows = self.config["data"]["max_rows"]
        self.seed = self.config["model"]["seed"]
        self.feature_names = self.config["features"]

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config was not found: {path}")

        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def read_data(self) -> DataFrame:
        self.log.info("Reading dataset: %s", self.input_path)

        data = self.spark.read.parquet(self.input_path)

        self.log.info("Input partitions: %d", data.rdd.getNumPartitions())
        return data

    @staticmethod
    def extract_product_information(
            data: DataFrame,
    ) -> DataFrame:
        product_name = (
            F.try_element_at(
                "product_name",
                F.lit(1),
            )
            .getField("text")
            .alias("product_name")
        )

        category = F.try_element_at(
            "categories_tags",
            F.lit(-1),
        )

        food_group = F.try_element_at(
            "food_groups_tags",
            F.lit(-1),
        )

        return (
            data
            .select(
                "code",
                product_name,
                "brands",
                "nutriscore_grade",
                category.alias("category"),
                food_group.alias("food_group"),
            )
            .dropDuplicates(["code"])
        )

    def extract_nutriments(self, data: DataFrame) -> DataFrame:
        nutriments = (
            data
            .select(
                "code",
                F.explode_outer("nutriments").alias("nutriment"),
            )
            .select(
                "code",
                F.col("nutriment.name").alias("name"),
                F.col("nutriment")
                .getField("100g")
                .cast("double")
                .alias("value_100g"),
            )
            .where(F.col("name").isin(self.feature_names))
            .where(F.col("value_100g").isNotNull())
        )

        return (
            nutriments
            .groupBy("code")
            .pivot("name", self.feature_names)
            .agg(F.first("value_100g", ignorenulls=True))
        )

    def filter_invalid_values(self, data: DataFrame) -> DataFrame:
        maximum_values = {
            "energy-kcal": 1000.0,
            "fat": 100.0,
            "carbohydrates": 100.0,
            "proteins": 100.0,
            "sugars": 100.0,
        }

        conditions = [
            F.col(feature).between(
                0.0, maximum_values[feature]
            ) for feature in self.feature_names
        ]

        combined_condition = reduce(
            lambda left, right: left & right, conditions
        )

        return (
            data.dropna(subset=["code", "product_name", *self.feature_names]).where(combined_condition)
        )

    def create_sample(self, data: DataFrame, row_count: int) -> DataFrame:
        if row_count <= self.max_rows:
            return data

        fraction = min(1.0, self.max_rows / row_count * 1.2)

        return (
            data.sample(withReplacement=False, fraction=fraction, seed=self.seed).limit(self.max_rows)
        )

    def save_data(self, data: DataFrame) -> None:
        result = data.repartition(8)

        result.write.mode("overwrite").parquet(self.output_path)

        saved_count = self.spark.read.parquet(self.output_path).count()

        self.log.info("Saved products: %d", saved_count)
        self.log.info("Output path: %s", self.output_path)

    def run(self) -> None:
        raw_data = self.read_data()

        product_information = self.extract_product_information(raw_data)
        nutriments = self.extract_nutriments(raw_data)

        prepared_data = nutriments.join(
            product_information, on="code", how="inner"
        )
        clean_data = self.filter_invalid_values(prepared_data)
        clean_data.persist(StorageLevel.MEMORY_AND_DISK)

        try:
            clean_count = clean_data.count()
            self.log.info("Valid products: %d", clean_count)

            result = self.create_sample(
                clean_data,
                clean_count
            )

            self.save_data(result)

        finally:
            clean_data.unpersist()


def main() -> None:
    spark = SparkSessionFactory.create(app_name="OpenFoodFactsProcessing")

    try:
        preprocessor = OpenFoodFactsPreprocessor(spark=spark)
        preprocessor.run()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

