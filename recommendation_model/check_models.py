import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

def build_items_lookup(ratings_df):
    # If you have a books table with titles:
    HAS_BOOKS = True  # Set to True if you have books_df with ISBN -> Book_Title
    BOOK_TITLE_COL = "Book_Title"  # Adjust if the column name is different
    books_df = spark.read.csv(
        r"big_data_recommendation_system\data\BX_Book.csv",
        header=True, inferSchema=True
    )
    
    # Use ISBN + itemId_int, and join with Book_Title if available
    base = ratings_df.select("ISBN", "itemId_int").dropDuplicates()
    if HAS_BOOKS:
        # books_df must exist in the session (ISBN, Book_Title)
        lookup = base.join(
            books_df.select("ISBN", BOOK_TITLE_COL).withColumnRenamed(BOOK_TITLE_COL, "Book_Title"),
            on="ISBN", how="left"
        )
    else:
        lookup = base.withColumn("Book_Title", F.col("ISBN").cast("string"))
    return lookup

# Function to create a histogram
def make_hist(df, rmse_col):
    return df.withColumn(
        "bin", (F.col(rmse_col) / 0.25).cast("int") * 0.25
    ).groupBy("bin").count().withColumnRenamed("count", "N")


spark = SparkSession.builder.appName("BookRecommendation").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

V_RMSE = {"UBCF": [], "IBCF": []}
ratings_df = spark.read.parquet(r"artifacts\full_ui_matrix.parquet")

# Lookup table for item names
items_lookup_df = build_items_lookup(ratings_df).cache()
print("after lookup")

train_path = "train_fold_{}.parquet"
test_path = "test_fold_{}.parquet"

# Create a list of (train, test) datasets
train_test_sets = []
num_folds = 5  # Number of folds

for i in range(1, num_folds + 1):
    # Read train and test files for each fold
    train = spark.read.parquet(train_path.format(i))
    test = spark.read.parquet(test_path.format(i))
    
    train_test_sets.append((train, test))


models = []  # List to store models

# Load saved models for each fold
for i in range(1, num_folds + 1):
    model_path = f"ALS_model_fold_{i}/"
    model = ALSModel.load(model_path)
    models.append(model)


# Basic RMSE evaluator
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="Book_Rating_normalized",
    predictionCol="prediction"
)

# Collect results for each model
results = []

for model_idx, model in enumerate(models, start=1):
    ub_rmses = []
    ib_rmses = []

    for _, test_df in train_test_sets:
        # Make predictions
        predictions = model.transform(test_df).na.drop(subset=["prediction"])

        # UB RMSE – average per user
        user_rmse_df = predictions.withColumn(
            "sq_error",
            F.pow(F.col("Book_Rating_normalized") - F.col("prediction"), 2)
        ).groupBy("User_ID").agg(F.sqrt(F.avg("sq_error")).alias("rmse_user"))
        ub_rmse = user_rmse_df.agg(F.avg("rmse_user")).first()[0]
        ub_rmses.append(ub_rmse)

        # IB RMSE – average per item
        item_rmse_df = predictions.withColumn(
            "sq_error",
            F.pow(F.col("Book_Rating_normalized") - F.col("prediction"), 2)
        ).groupBy("itemId_int").agg(F.sqrt(F.avg("sq_error")).alias("rmse_item"))
        ib_rmse = item_rmse_df.agg(F.avg("rmse_item")).first()[0]
        ib_rmses.append(ib_rmse)

    # Average over all test sets
    ub_avg = sum(ub_rmses) / len(ub_rmses)
    ib_avg = sum(ib_rmses) / len(ib_rmses)

    results.append({
        "model": model_idx,
        "ub_rmse_avg": ub_avg,
        "ib_rmse_avg": ib_avg
    })


# Print summary
results_df = pd.DataFrame(results)
print("Average RMSE for each model (UB and IB):")
print(results_df)

# Optionally, select the best model for each method
best_ub_model = results_df.sort_values("ub_rmse_avg").iloc[0]["model"]
best_ib_model = results_df.sort_values("ib_rmse_avg").iloc[0]["model"]

print(f"Best model for UB: ALS_model_fold_{int(best_ub_model)}")
print(f"Best model for IB: ALS_model_fold_{int(best_ib_model)}")
