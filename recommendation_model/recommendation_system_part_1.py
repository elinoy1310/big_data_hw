from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.storagelevel import StorageLevel

# =======================
# Create SparkSession
# =======================
spark = SparkSession.builder.appName("BookRecommendation").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# =======================
# Load CSV
# =======================
ratings_df = spark.read.csv(
    r"big_data_recommendation_system\data\BX_Book_Ratings_clean.csv",
    header=True, inferSchema=True
)

# =======================
# Map ISBN to small numeric IDs
# =======================
isbn_indexer = StringIndexer(inputCol="ISBN", outputCol="itemId_int", handleInvalid="skip")
ratings_df = isbn_indexer.fit(ratings_df).transform(ratings_df)

# =======================
# Compute average rating per user for normalization
# =======================
user_avg_window = Window.partitionBy("User_ID")
ratings_df_mapped = ratings_df.withColumn(
    "user_mean",
    F.avg("Book_Rating").over(user_avg_window)
)

ratings_df = ratings_df_mapped.withColumn(
    "Book_Rating_normalized",
    F.col("Book_Rating") - F.col("user_mean")
)

# =======================
# Prepare 5 k-fold sets and persist in memory + disk
# =======================
k = 5
train_test_sets = []

for i in range(k):
    # Create Train/Test split
    train, test = ratings_df.randomSplit([0.8, 0.2], seed=42+i)
    
    # Persist in memory
    train.cache()
    test.cache()

    train.persist(StorageLevel.MEMORY_AND_DISK)
    test.persist(StorageLevel.MEMORY_AND_DISK)

    # Add to list
    train_test_sets.append((train, test))
    
    # Also save to disk as Parquet
    train.write.mode("overwrite").parquet(f"big_data_recommendation_system/recommendation_model/artifacts_model/train_fold_{i+1}.parquet")
    test.write.mode("overwrite").parquet(f"big_data_recommendation_system/recommendation_model/artifacts_model/test_fold_{i+1}.parquet")

# =======================
# Function to train ALS and compute RMSE
# =======================
def train_als(train_df, test_df, rank=10, maxIter=10, regParam=0.1):
    als = ALS(
        userCol="User_ID",        
        itemCol="itemId_int",          # ISBN converted to numeric ID
        ratingCol="Book_Rating_normalized",  # Use normalized rating
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        coldStartStrategy="drop",
        nonnegative=True
    )

    model = als.fit(train_df)
    predictions = model.transform(test_df)
    
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="Book_Rating_normalized",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return model, rmse

# =======================
# Train on all folds
# =======================
rmse_list = []
models = []

for i, (train, test) in enumerate(train_test_sets):
    print(f"Training k-fold {i+1}")
    model, rmse = train_als(train, test)
    rmse_list.append(rmse)
    models.append(model)
    print(f"RMSE k-fold {i+1}: {rmse}")

# =======================
# Save all models
# =======================
for i, model in enumerate(models):
    path = f"big_data_recommendation_system/recommendation_model/artifacts_model/ALS_model_fold_{i+1}"
    model.write().overwrite().save(path)
