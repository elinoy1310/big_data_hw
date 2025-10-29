import os
import time
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.window import Window


LOG_PATH = "run_log.txt"

def log(message):
    """Log message to both console and log file with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def build_items_lookup(ratings_df, spark):
    """Build items lookup efficiently"""
    log("Building items lookup table...")

    base = ratings_df.select("ISBN", "itemId_int").dropDuplicates().coalesce(1)

    try:
        log("Opening BX_Book.csv...")
        books_df = spark.read.csv(
            r"big_data_recommendation_system\data\BX_Book.csv",
            header=True, inferSchema=True
        )
        log(f"BX_Book.csv loaded successfully. Columns: {books_df.columns}")
        lookup = base.join(
            books_df.select("ISBN", "Book_Title").coalesce(1),
            on="ISBN",
            how="left"
        ).fillna({"Book_Title": "Unknown"})
        log("Book titles successfully linked to lookup table.")
    except Exception as e:
        log(f"Failed to load BX_Book.csv ({e}). Using ISBN as title.")
        lookup = base.withColumn("Book_Title", F.col("ISBN").cast("string"))

    return lookup


def measure_sparsity(ratings_df):
    """Compute and log sparsity metrics for the ratings matrix."""
    log("Calculating sparsity and dataset structure...")
    num_users = ratings_df.select("User_ID").distinct().count()
    num_items = ratings_df.select("itemId_int").distinct().count()
    num_ratings = ratings_df.count()
    sparsity = 1 - (num_ratings / (num_users * num_items))
    log(f"Users: {num_users}, Items: {num_items}, Ratings: {num_ratings}")
    log(f"Matrix sparsity: {sparsity:.6f}")
    return num_users, num_items, num_ratings, sparsity


def time_stage(func, stage_name, *args, **kwargs):
    """Run a function with timing and logging."""
    log(f"--- Starting stage: {stage_name} ---")
    start = time.time()
    result = func(*args, **kwargs)
    duration = (time.time() - start) / 60
    log(f"--- Finished stage: {stage_name} in {duration:.2f} minutes ---")
    return result


def get_ubcf_recommendations(model, ratings_df, items_lookup_df, num_users=5):
    """Generate UBCF recommendations (user-based)."""
    from pyspark.sql import Row
    log(f"Generating UBCF recommendations for {num_users} users...")

    random_users = (ratings_df.select("User_ID").distinct()
                    .orderBy(F.rand())
                    .limit(num_users)
                    .coalesce(1))
    user_list = [r.User_ID for r in random_users.collect()]
    user_subset_df = spark.createDataFrame([Row(User_ID=u) for u in user_list])

    top10_ub = model.recommendForUserSubset(user_subset_df, 10)
    log("ALS model produced raw recommendations (UBCF).")

    top10_ub_exploded = (top10_ub
                         .select("User_ID", F.explode("recommendations").alias("rec"))
                         .select("User_ID",
                                 F.col("rec.itemId_int").alias("itemId_int"),
                                 F.col("rec.rating").alias("score")))

    top10_ub_with_titles = top10_ub_exploded.join(items_lookup_df, on="itemId_int", how="left")
    window = Window.partitionBy("User_ID").orderBy(F.col("score").desc())

    ranked = (top10_ub_with_titles
              .withColumn("rn", F.row_number().over(window))
              .filter(F.col("rn") <= 10)
              .withColumn("short_name", F.substring("Book_Title", 1, 12)))

    recommendations = []
    for user_id in user_list:
        user_recs = ranked.filter(F.col("User_ID") == user_id).orderBy("rn").collect()
        books = [r.short_name if r.short_name else f"Item{r.itemId_int}" for r in user_recs]
        while len(books) < 10:
            books.append("")
        recommendations.append({"user": user_id, "books": "  ".join(books[:10])})

    log(f"Finished generating UBCF recommendations for {num_users} users.")
    return recommendations


def get_ibcf_recommendations(model, ratings_df, items_lookup_df, num_users=5):
    """Generate IBCF recommendations (item-based)."""
    from pyspark.sql import Row
    log(f"Generating IBCF recommendations for {num_users} users...")

    random_users = (ratings_df.select("User_ID").distinct()
                    .orderBy(F.rand())
                    .limit(num_users)
                    .coalesce(1))
    user_list = [r.User_ID for r in random_users.collect()]
    user_subset_df = spark.createDataFrame([Row(User_ID=u) for u in user_list])

    top10_ib = model.recommendForUserSubset(user_subset_df, 10)
    log("ALS model produced raw recommendations (IBCF).")

    top10_ib_exploded = (top10_ib
                         .select("User_ID", F.explode("recommendations").alias("rec"))
                         .select("User_ID",
                                 F.col("rec.itemId_int").alias("itemId_int"),
                                 F.col("rec.rating").alias("score")))

    joined = top10_ib_exploded.join(items_lookup_df, on="itemId_int", how="left")
    window = Window.partitionBy("User_ID").orderBy(F.col("score").desc())

    ranked = (joined
              .withColumn("rn", F.row_number().over(window))
              .filter(F.col("rn") <= 10)
              .withColumn("short_name", F.substring("Book_Title", 1, 12)))

    recommendations = []
    for user_id in user_list:
        user_recs = ranked.filter(F.col("User_ID") == user_id).orderBy("rn").collect()
        books = [r.short_name if r.short_name else f"Item{r.itemId_int}" for r in user_recs]
        while len(books) < 10:
            books.append("")
        recommendations.append({"user": user_id, "books": "  ".join(books[:10])})

    log(f"Finished generating IBCF recommendations for {num_users} users.")
    return recommendations


# ----------------------------- MAIN EXECUTION -----------------------------
if __name__ == "__main__":
    # Start new log
    open(LOG_PATH, "w").close()
    log("===== BEGIN RECOMMENDER SYSTEM RUN =====")

    start_all = time.time()

    # Initialize Spark
    spark = (
        SparkSession.builder
        .appName("recommendation_system_with_logging")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log("Spark session initialized.")

    # Load data
    log("Loading ratings parquet file...")
    ratings_df = spark.read.parquet(
        r"big_data_recommendation_system\recommendation_model\artifacts_model\full_ui_matrix.parquet"
    )
    log("Ratings file loaded successfully.")
    measure_sparsity(ratings_df)

    # Build lookup
    items_lookup_df = time_stage(build_items_lookup, "Build Items Lookup", ratings_df, spark).cache()
    # Load models
    model_path_ub = r"big_data_recommendation_system\recommendation_model\artifacts_model\ALS_model_fold_2"
    model_path_ib = r"big_data_recommendation_system\recommendation_model\artifacts_model\ALS_model_fold_4"
    log(f"Loading UBCF model from {model_path_ub}...")
    model_ub = ALSModel.load(model_path_ub)
    log("UBCF model loaded.")
    log(f"Loading IBCF model from {model_path_ib}...")
    model_ib = ALSModel.load(model_path_ib)
    log("IBCF model loaded.")

    num_recommendation_users = 250
    # Generate recommendations with timing
    ub_recs = time_stage(get_ubcf_recommendations, "UBCF Recommendations", model_ub, ratings_df, items_lookup_df, num_recommendation_users)
    ib_recs = time_stage(get_ibcf_recommendations, "IBCF Recommendations", model_ib, ratings_df, items_lookup_df, num_recommendation_users)

    # Summary
    total_time = (time.time() - start_all) / 60
    log(f"Total run time: {total_time:.2f} minutes.")
    log("===== END RECOMMENDER SYSTEM RUN =====")

    # Clean up
    items_lookup_df.unpersist()
    spark.stop()
