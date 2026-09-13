from pyspark.sql import DataFrame, SparkSession
from logger import Logger

class CassandraStorage:
    def __init__(
        self,
        spark: SparkSession,
        keyspace: str,
        source_table: str,
        result_table: str
    ) -> None:
        self.spark = spark
        self.keyspace = keyspace
        self.source_table = source_table
        self.result_table = result_table

        logger = Logger(show=True)
        self.log = logger.get_logger(__name__)

    def read_products(self) -> DataFrame:
        self.log.info("Reading data from Cassandra: %s.%s", self.keyspace, self.source_table)

        data = (
            self.spark.read
            .format("org.apache.spark.sql.cassandra")
            .options(
                keyspace=self.keyspace,
                table=self.source_table,
            )
            .load()
        )

        self.log.info("Input partitions: %d", data.rdd.getNumPartitions())
        return data

    def write_predictions(self, data: DataFrame) -> None:
        self.log.info("Writing predictions to Cassandra: %s.%s", self.keyspace, self.result_table)

        (
            data.write
            .format("org.apache.spark.sql.cassandra")
            .mode("append")
            .options(
                keyspace=self.keyspace,
                table=self.result_table
            )
            .save()
        )

        self.log.info("Predictions successfully saved to Cassandra")