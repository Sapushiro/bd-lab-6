from pyspark.sql import DataFrame, SparkSession

from logger import Logger
from spark_session import SparkSessionFactory

from pathlib import Path
import json

class CassandraDataLoader:
    def __init__(
        self,
        spark: SparkSession,
        config_path: str="config.json"
    ) -> None:
        self.spark = spark
        self.config = self._load_config(config_path)
        logger = Logger(show=True)
        self.log = logger.get_logger(__name__)

        self.input_path = self.config["data"]["output_path"]
        cassandra_config = self.config["cassandra"]
        self.keyspace = cassandra_config["keyspace"]
        self.table = cassandra_config["source_table"]

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Config was not found: {path}"
            )

        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def read_data(self) -> DataFrame:
        self.log.info("Reading prepared dataset: %s", self.input_path)
        data = self.spark.read.parquet(self.input_path)

        self.log.info("Rows to upload: %d", data.count())
        return data

    @staticmethod
    def prepare_data(data: DataFrame) -> DataFrame:
        return (
            data
            .withColumnRenamed("energy-kcal", "energy_kcal")
        )

    def write_data(self, data: DataFrame) -> None:
        self.log.info("Writing data to Cassandra: %s.%s", self.keyspace, self.table)

        (
            data.write
            .format("org.apache.spark.sql.cassandra")
            .mode("append")
            .options(
                keyspace=self.keyspace,
                table=self.table
            )
            .save()
         )

        self.log.info("Data successfully uploaded to Cassandra")

    def run(self) -> None:
        data = self.read_data()
        prepared_data = self.prepare_data(data)
        self.write_data(prepared_data)


def main() -> None:
    spark = SparkSessionFactory.create(
        app_name="OpenFoodFactsCassandraLoader"
    )

    try:
        loader = CassandraDataLoader(spark=spark)
        loader.run()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()