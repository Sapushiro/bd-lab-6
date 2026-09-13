from pyspark.sql import SparkSession

def main() -> None:
    spark = (
        SparkSession.builder
        .appName("WordCount")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print("\nSpark configuration:")
    print(f"Spark version: {spark.version}")
    print(f"Application name: {spark.sparkContext.appName}")
    print(f"Master: {spark.sparkContext.master}")
    print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")
    print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")
    print(
        "Shuffle partitions: "
        f"{spark.conf.get('spark.sql.shuffle.partitions')}"
    )

    lines = spark.sparkContext.textFile("test/text.txt")
    word_counts = (
        lines
        .flatMap(lambda line: line.lower().split())
        .map(lambda word: (word, 1))
        .reduceByKey(lambda left, right: left + right)
        .sortBy(lambda item: (-item[1], item[0]))
    )

    print("\nWord count result:")
    for word, count in word_counts.collect():
        print(f"{word}: {count}")

    spark.stop()


if __name__ == "__main__":
    main()