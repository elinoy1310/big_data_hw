import os
import json
import pickle

# ============
# CONFIG
# ============
import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.feature import Normalizer, BucketedRandomProjectionLSH

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.storagelevel import StorageLevel
from pyspark.storagelevel import StorageLevel

from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, sqrt, avg

# =======================
# יצירת SparkSession
# =======================
spark = SparkSession.builder.appName("BookRecommendation").master("local[*]").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

save_dir = r"big_data_recommendation_system\recommendation_model\artifacts_model"
os.makedirs(save_dir, exist_ok=True)

# ---------------------------
# 1. UI (ratings_df)
# Save the user-item ratings DataFrame
# ---------------------------

ui_cols = ["User_ID", "ISBN", "Book_Rating"]

ratings_df = spark.read.csv(
    r"big_data_recommendation_system\data\BX_Book_Ratings_clean.csv",
    header=True, inferSchema=True
)

ratings_df.select(ui_cols).write.mode("overwrite").parquet(os.path.join(save_dir, "UI.parquet"))

# ---------------------------
# 2. eval_sets (train/test folds)
# Save each train/test split as Parquet files
# ---------------------------
# <already done in recommendation_system.py>


# ---------------------------
# 3. R.UB, R.IB (ALS models)
# the models were talen after running big_data_recommendation_system\recommendation_model\check_models.py
# Save both ALS models
# ---------------------------
model_path = r"big_data_recommendation_system\recommendation_model\artifacts_model\ALS_model_fold_2"  
model = ALSModel.load(model_path)
model.write().overwrite().save(os.path.join(save_dir, "R_UB_model"))

model_path = r"big_data_recommendation_system\recommendation_model\artifacts_model\ALS_model_fold_4"  
model = ALSModel.load(model_path)
model.write().overwrite().save(os.path.join(save_dir, "R_IB_model"))

# ---------------------------
# 4. V.RMSE (RMSE per user)
# V_RMSE is defined as a dict: {"UBCF": [...], "IBCF": [...]}
# ---------------------------


# =======================
# Paths to the ALS models (UBCF and IBCF)
# =======================
R_UB_model_path = os.path.join(save_dir, "R_UB_model")
R_IB_model_path = os.path.join(save_dir, "R_IB_model")

model_UB = ALSModel.load(R_UB_model_path)
model_IB = ALSModel.load(R_IB_model_path)

# =======================
# Function to compute RMSE per user
# =======================
def compute_user_rmse(model, test_df):
    preds = model.transform(test_df).dropna()
    user_rmse_df = preds.withColumn(
        "sq_error", (col("Book_Rating") - col("prediction"))**2
    ).groupBy("User_ID").agg(sqrt(avg("sq_error")).alias("rmse"))
    # Convert to dictionary: {user_id: rmse}
    user_rmse = {row['User_ID']: row['rmse'] for row in user_rmse_df.collect()}
    return user_rmse

# =======================
# Select one of the test folds for evaluation (2/4 (according to the models))
# =======================

test_fold_UB = spark.read.parquet(r"big_data_recommendation_system\recommendation_model\artifacts_model\test_fold_2.parquet")
test_fold_IB = spark.read.parquet(r"big_data_recommendation_system\recommendation_model\artifacts_model\test_fold_4.parquet")

# =======================
# Compute V.RMSE
# =======================
V_RMSE = {
    'UBCF': compute_user_rmse(model_UB, test_fold_UB),
    'IBCF': compute_user_rmse(model_IB, test_fold_IB)
}

# =======================
# Save the entire "model.rdata" equivalent as a single Pickle file
# =======================
all_data = {
    'UI': os.path.join(save_dir, "UI.parquet"),
    'eval_sets': [  
        {
            'train': os.path.join(save_dir, f"train_fold_{i+1}.parquet"),
            'test': os.path.join(save_dir, f"test_fold_{i+1}.parquet")
        } for i in range(5)
    ],
    'R.UB': R_UB_model_path,
    'R.IB': R_IB_model_path,
    'V.RMSE': V_RMSE
}

with open(os.path.join(save_dir, "model.pkl"), "wb") as f:
    pickle.dump(all_data, f)

print("Saved model.pkl with UI, eval_sets, R.UB, R.IB and V.RMSE")
