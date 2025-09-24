import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F


def build_items_lookup(ratings_df):
    # If you have a books table with titles:
    HAS_BOOKS = True  # change to True if you have books_df with ISBN -> Book_Title
    BOOK_TITLE_COL = "Book_Title"  # adjust if different
    books_df = spark.read.csv(
    r"big_data_recommendation_system\data\BX_Book.csv",
    header=True, inferSchema=True
)
    
    # We'll use ISBN + itemId_int, and if Book_Title exists — join to it
    base = ratings_df.select("ISBN", "itemId_int").dropDuplicates()
    if HAS_BOOKS:
        # books_df should exist in the session (ISBN, Book_Title)
        # adjust if table name is different
        lookup = base.join(books_df.select("ISBN", BOOK_TITLE_COL).withColumnRenamed(BOOK_TITLE_COL, "Book_Title"), on="ISBN", how="left")
    else:
        lookup = base.withColumn("Book_Title", F.col("ISBN").cast("string"))
    return lookup

# function to prepare histogram
def make_hist(df, rmse_col):
    return df.withColumn(
        "bin", (F.col(rmse_col) / 0.25).cast("int") * 0.25
    ).groupBy("bin").count().withColumnRenamed("count", "N")

def write_model_txt(ub_rmse_avg, ib_rmse_avg, ub_hist_df, ib_hist_df, top10_ub, top10_ib, link_to_model_json=None, path="model.txt"):
    # build a unified histogram table with standard bins
    bins = [x * 0.25 for x in range(0, 21)]
    bins_df = spark.createDataFrame([(float(b),) for b in bins], ["RMSE_bin"])
    ub = ub_hist_df.select(F.col("bin").alias("RMSE_bin"), F.col("N").alias("N_UBCF"))
    ib = ib_hist_df.select(F.col("bin").alias("RMSE_bin"), F.col("N").alias("N_IBCF"))
    hist = bins_df.join(ub, on="RMSE_bin", how="left").join(ib, on="RMSE_bin", how="left")\
                  .na.fill({"N_UBCF": 0, "N_IBCF": 0})\
                  .orderBy("RMSE_bin")

    # collect to Python to write txt
    hist_rows = hist.collect()

    # prepare Top-10 in table format (user, book1..book10) for both methods
    def pivot_top10(top10_df):
        # top10_df: (User_ID, itemId_int, score, short_name, rn)
        # convert to one row per user with 10 columns
        pvt = top10_df.select("User_ID", "short_name", "rn")\
                      .groupBy("User_ID")\
                      .pivot("rn", list(range(1, 11)))\
                      .agg(F.first("short_name"))\
                      .orderBy("User_ID")
        return pvt
    
    ub_table = pivot_top10(top10_ub).collect()
    ib_table = pivot_top10(top10_ib).collect()
    
    # writing
    with open(path, "w", encoding="utf-8") as f:
        f.write("-------------------- BEGIN model.txt --------------------\n")
        f.write("# Team: big team\n")
        f.write("# Date: 15.9.2025\n")
        f.write("# Database name   books2.db\n")
        f.write("5) link to model.rdata {}\n".format(link_to_model_json))
        f.write("\n")
        f.write("6.a) RMSE of the full model UB {:.4f}, IB {:.4f}\n".format(ub_rmse_avg, ib_rmse_avg))
        f.write("6.b) histogram of RMSE <table(bin, num occurrences UBCF, num occurrences IBCF)>\n")
        f.write("RMSE\n")
        f.write("      N.UBCF   N.IBCF\n")
        for r in hist_rows:
            f.write("{:<4.2f}  {:<7d}  {:<7d}\n".format(r["RMSE_bin"], int(r["N_UBCF"]), int(r["N_IBCF"])))
        f.write("6.c) Top-10 recommendations   <table(user,book1,book2,book3, ..., book10)>\n")
        # UBCF
        f.write("UBCF\n")
        f.write("user\n")
        for row in ub_table:
            user = row["User_ID"]
            books = [row.get(c) if row.get(c) is not None else "" for c in range(1, 11)]
            f.write("{:<7} {}\n".format(str(user), "  ".join([str(b)[:12] for b in books])))
        f.write("\nIBCF\n")
        f.write("user\n")
        for row in ib_table:
            user = row["User_ID"]
            books = [row.get(c) if row.get(c) is not None else "" for c in range(1, 11)]
            f.write("{:<7} {}\n".format(str(user), "  ".join([str(b)[:12] for b in books])))
        f.write("...\n")
        f.write("-------------------- END model.txt --------------------\n")

if __name__=="__main__":
    spark = SparkSession.builder.appName("BookRecommendation").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    V_RMSE = {"UBCF": [], "IBCF": []}
    ratings_df=spark.read.parquet(r"big_data_recommendation_system\recommendation_model\artifacts_model\full_ui_matrix.parquet")

    items_lookup_df = build_items_lookup(ratings_df).cache()
    print("after lookup")

    train_path = "train_fold_{}.parquet"
    test_path = "test_fold_{}.parquet"

    # create a list of (train, test) from files
    train_test_sets = []
    num_folds = 5  # number of folds 

    for i in range(1, num_folds + 1):
        # read train and test files for each fold
        train = spark.read.parquet(train_path.format(i))
        test = spark.read.parquet(test_path.format(i))
        
        # add to list
        train_test_sets.append((train, test))


    from pyspark.ml.recommendation import ALSModel
    models = []  # list to hold models

    # load saved models for each fold
    for i in range(1, num_folds + 1):
        # load the model for each fold from .parquet folder
        model_path = f"big_data_recommendation_system/recommendation_model/artifacts_model/ALS_model_fold_{i}/"  # path to model per fold
        model = ALSModel.load(model_path)
        
        # add model to list
        models.append(model)

    # lists to store averages for each fold
    ub_rmse_per_fold = []
    ib_rmse_per_fold = []
    ub_hist_dfs = []
    ib_hist_dfs = []

    # evaluator for RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="Book_Rating_normalized",
        predictionCol="prediction"
    )

    # for each fold and corresponding model
    for i, ((train_df, test_df), model) in enumerate(zip(train_test_sets, models), start=1):
        # prediction on test set
        predictions = model.transform(test_df)
        
        # drop missing values
        predictions = predictions.na.drop(subset=["prediction"])
        
        # compute RMSE per user
        user_rmse_df = predictions.withColumn(
            "sq_error",
            F.pow(F.col("Book_Rating_normalized") - F.col("prediction"), 2)
        ).groupBy("User_ID").agg(F.sqrt(F.avg("sq_error")).alias("rmse_user"))

         # compute RMSE per item
        item_rmse_df = predictions.withColumn(
            "sq_error",
            F.pow(F.col("Book_Rating_normalized") - F.col("prediction"), 2)
        ).groupBy("itemId_int").agg(F.sqrt(F.avg("sq_error")).alias("rmse_item"))

        # average RMSE over all users
        avg_rmse = user_rmse_df.agg(F.avg("rmse_user")).first()[0]
        ub_rmse_per_fold.append(avg_rmse)

        # average RMSE over all items
        it_avg_rmse = item_rmse_df.agg(F.avg("rmse_item")).first()[0]
        ib_rmse_per_fold.append(it_avg_rmse)
        
        print(f"Fold {i} - avg RMSE: {avg_rmse:.4f}")
        print(f"Fold {i} - IBCF avg RMSE: {it_avg_rmse:.4f}")

        # add fold column to know from which fold
        user_rmse_df = user_rmse_df.withColumn("fold", F.lit(i))
        
        ub_hist_dfs.append(user_rmse_df)

        # after computing item_rmse_df, add fold column
        item_rmse_df = item_rmse_df.withColumn("fold", F.lit(i))
        ib_hist_dfs.append(item_rmse_df)

    # overall average across all folds
    ub_rmse_avg = sum(ub_rmse_per_fold) / len(ub_rmse_per_fold)
    print(f"UB RMSE avg over all folds: {ub_rmse_avg:.4f}")
    ib_rmse_avg = sum(ib_rmse_per_fold) / len(ib_rmse_per_fold)
    print(f"IB RMSE avg over all folds: {ib_rmse_avg:.4f}")
    # union all folds into one table
    ub_hist_df = ub_hist_dfs[0]
    for df in ub_hist_dfs[1:]:
        ub_hist_df = ub_hist_df.union(df)

    # can save to CSV or Parquet
    ub_hist_df = make_hist(ub_hist_df, "rmse_user")
    #ub_hist_df.write.mode("overwrite").parquet("ub_hist_df.parquet")

    print("UB histogram DataFrame saved to ub_hist_df.parquet")

    ib_hist_df = ib_hist_dfs[0]
    for df in ib_hist_dfs[1:]:
        ib_hist_df = ib_hist_df.union(df)

    # save to parquet
    ib_hist_df = make_hist(ib_hist_df, "rmse_item")
    the_best_model=models[1]
    # select 500 random users
    random_users = ratings_df.select("User_ID").distinct().orderBy(F.rand()).limit(5)

    top10_ub = the_best_model.recommendForAllUsers(10)
    top10_ub = top10_ub.join(random_users, on="User_ID", how="inner")
    print("Top 10 recommendations for 500 random users generated.")

    top10_ub = top10_ub.select(
        "User_ID",
        F.explode("recommendations").alias("rec")
    ).select(
        "User_ID",
        F.col("rec.itemId_int").alias("itemId_int"),
        F.col("rec.rating").alias("score")
    )
    print("Exploded recommendations into individual rows.")

    top10_ub = top10_ub.join(items_lookup_df, on="itemId_int", how="left")
    print("Joined with items lookup to get book titles.")

    window = Window.partitionBy("User_ID").orderBy(F.col("score").desc())
    top10_ub = top10_ub.withColumn("rn", F.row_number().over(window)).filter(F.col("rn") <= 10)
    print("Filtered to top 10 recommendations per user.")

    # shorten name to 12 chars
    top10_ub = top10_ub.withColumn("short_name", F.substring("Book_Title", 1, 12))
    #top10_ub.write.mode("overwrite").parquet("top10_ub.parquet")

    print("top 10 ub saved to top10_ub.parquet")

    top10_ib = models[3].recommendForAllItems(10)

    top10_ib = top10_ib.select(
        "itemId_int",
        F.explode("recommendations").alias("rec")
    ).select(
        "itemId_int",
        F.col("rec.User_ID").alias("User_ID"),
        F.col("rec.rating").alias("score")
    )
    print("Exploded IBCF recommendations into individual rows.")

    # filter to the 500 selected users
    top10_ib = top10_ib.join(random_users, on="User_ID", how="inner")

    top10_ib = top10_ib.join(items_lookup_df, on="itemId_int", how="left")
    print("Joined IBCF with items lookup to get book titles.")

    window_ib = Window.partitionBy("User_ID").orderBy(F.col("score").desc())
    top10_ib = top10_ib.withColumn("rn", F.row_number().over(window_ib)).filter(F.col("rn") <= 10)

    # shorten name to 12 chars
    top10_ib = top10_ib.withColumn("short_name", F.substring("Book_Title", 1, 12))
    print("Filtered IBCF to top 10 recommendations per user.")

    write_model_txt(
        ub_rmse_avg, ib_rmse_avg,
        ub_hist_df, ib_hist_df,
        top10_ub, top10_ib
    )
