import json
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, StandardScalerModel, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

from spark_session import SparkSessionFactory
from cassandra_storage import CassandraStorage

from logger import Logger

class KMeansTrainer:
    def __init__(self, spark: SparkSession, config_path: str="config.json"):
        self.spark = spark
        self.config = self._load_config(config_path)

        logger = Logger(show=True)
        self.log = logger.get_logger(__name__)

        self.feature_names = self.config["features"]
        model_config = self.config["model"]
        self.model_path = model_config["output_path"]
        self.metrics_path = model_config["metrics_path"]
        self.predictions_path = model_config["predictions_path"]

        self.min_k = model_config["min_k"]
        self.max_k = model_config["max_k"]
        self.seed = model_config["seed"]
        self.max_iter = model_config["max_iter"]
        self.tolerance = model_config["tolerance"]

        cassandra_config = self.config["cassandra"]
        self.storage = CassandraStorage(
            spark=self.spark,
            keyspace=cassandra_config["keyspace"],
            source_table=cassandra_config["source_table"],
            result_table=cassandra_config["result_table"],
        )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config was not found: {path}")

        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def read_data(self) -> DataFrame:
        data = self.storage.read_products()
        data = data.withColumnRenamed("energy_kcal", "energy-kcal")

        self.log.info("Input partitions: %d", data.rdd.getNumPartitions())
        return data

    def prepare_features(self, data: DataFrame) -> tuple[DataFrame, VectorAssembler, StandardScalerModel]:
        assembler = VectorAssembler(
            inputCols=self.feature_names,
            outputCol="raw_features",
            handleInvalid="error"
        )

        assembled_data = assembler.transform(data)

        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withMean=True,
            withStd=True
        )

        scaler_model = scaler.fit(assembled_data)
        scaled_data = scaler_model.transform(assembled_data)

        return scaled_data, assembler, scaler_model

    def train_candidates(self, scaled_data: DataFrame) -> tuple[KMeansModel, list[dict]]:
        evaluator = ClusteringEvaluator(
            featuresCol="scaled_features",
            predictionCol="prediction",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )

        best_model = None
        best_silhouette = float("-inf")
        metrics = []

        for k in range(self.min_k, self.max_k + 1):
            self.log.info("Training KMeans with k=%d", k)

            estimator = KMeans(
                k=k,
                seed=self.seed,
                featuresCol="scaled_features",
                predictionCol="prediction",
                maxIter=self.max_iter,
                tol=self.tolerance,
                distanceMeasure="euclidean"
            )

            model = estimator.fit(scaled_data)
            predictions = model.transform(scaled_data)

            silhouette = evaluator.evaluate(predictions)
            iterations = model.summary.numIter

            result = {
                "k": k,
                "silhouette": float(silhouette),
                "iterations": int(iterations),
            }

            metrics.append(result)

            self.log.info("Silhouette for k=%d: %.6f",k, silhouette)
            self.log.info("Iterations for k=%d: %d",k, iterations)

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_model = model

        if best_model is None:
            raise RuntimeError("KMeans model was not trained")

        self.log.info("Best model: k=%d, silhouette=%.6f",best_model.getK(),best_silhouette)
        return best_model, metrics

    def save_model(
            self,
            assembler: VectorAssembler,
            scaled_model: StandardScalerModel,
            kmeans_model: KMeansModel
    ) -> None:
        pipeline_model = PipelineModel(
            stages=[
                assembler,
                scaled_model,
                kmeans_model
            ]
        )
        pipeline_model.write().overwrite().save(self.model_path)
        self.log.info("Model saved: %s",self.model_path)

    def save_metrics(self, metrics: list[dict],) -> None:
        metrics_path = Path(self.metrics_path)
        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with metrics_path.open(mode="w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
            self.log.info("Metrics saved: %s",metrics_path)

    def save_predictions(self, predictions: DataFrame) -> None:
        columns_to_save = [
            "code",
            "product_name",
            "brands",
            "category",
            "food_group",
            "nutriscore_grade",
            *self.feature_names,
            "prediction",
        ]

        result = (
                predictions
                .select(*columns_to_save)
                .withColumnRenamed(
                    "energy-kcal",
                    "energy_kcal"
                )
        )
        self.storage.write_predictions(result)

    def run(self) -> None:
        data = self.read_data()

        scaled_data, assembler, scaled_model = self.prepare_features(data)
        scaled_data.persist(StorageLevel.MEMORY_AND_DISK)

        try:
            scaled_count = scaled_data.count()
            self.log.info("Prepared feature vectors: %d", scaled_count)

            best_model, metrics = self.train_candidates(scaled_data)

            self.save_model(assembler, scaled_model, best_model)
            self.save_metrics(metrics)

            predictions = best_model.transform(scaled_data)
            self.save_predictions(predictions)
        finally:
            scaled_data.unpersist()


def main() -> None:
    spark = SparkSessionFactory.create(app_name="OpenFoodFactsKMeans")
    try:
        trainer = KMeansTrainer(spark=spark)
        trainer.run()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()