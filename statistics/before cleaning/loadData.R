
# Loading packages
library(DBI)
library(RSQLite)

# Connecting to the SQLite database
con <- dbConnect(SQLite(), "books2.db")

# Collecting statistics

# How many users
num_users <- dbGetQuery(con, "SELECT COUNT(*) AS num_users FROM `BX_Users`;")$num_users

# How many books
num_books <- dbGetQuery(con, "SELECT COUNT(*) AS num_books FROM `BX_Books`;")$num_books

# How many ratings
num_ratings <- dbGetQuery(con, "SELECT COUNT(*) AS num_ratings FROM `BX_Book_Ratings`;")$num_ratings

# User rating histogram - how many ratings per user
user_hist <- dbGetQuery(con, "
  SELECT COUNT(*) AS num_ratings
  FROM `BX_Book_Ratings`
  WHERE `Book_Rating`>0
  GROUP BY `User_ID`
")

# Book rating histogram - how many ratings per book
book_hist <- dbGetQuery(con, "
  SELECT COUNT(*) AS num_ratings
  FROM `BX_Book_Ratings`
  WHERE `Book_Rating`>0
  GROUP BY ISBN
")

# Top rated books
top_books <- dbGetQuery(con, "
  SELECT b.`Book_Title`, b.`Book_Author`, COUNT(r.`Book_Rating`) AS num_ratings
  FROM `BX_Book_Ratings` r
  JOIN `BX_Books` b ON r.ISBN = b.ISBN
  WHERE `Book_Rating`>0
  GROUP BY r.ISBN
  ORDER BY num_ratings DESC
  LIMIT 10;
")

# Top active users
top_users <- dbGetQuery(con, "
  SELECT r.`User_ID`, COUNT(*) AS num_ratings
  FROM `BX_Book_Ratings` r
  WHERE `Book_Rating`>0
  GROUP BY r.`User_ID`
  ORDER BY num_ratings DESC
  LIMIT 10;
")

# Writing to the basics.txt file

writeLines(
  c(
    "-------------------- BEGIN basics.txt --------------------",
    "# Team: <team name>",
    "# Date: <date>",
    "# Database name   books2.db",
    
    # Part 3.a) How many users?
    paste("3.a) how many users?      ", num_users),
    
    # Part 3.b) How many books?
    paste("3.b) how many books?      ", num_books),
    
    # Part 3.c) How many ratings?
    paste("3.c) how many ratings?    ", num_ratings),
    
    # Part 3.d) Histogram of user ratings (how many users have rated N times?)
    "3.d) histogram of user-ratings <table(num ratings, num users)>",
    "(how many users have rated N times?)",
    "+-----+-------+",
    "| bin | N     |",
    "+-----+-------+",
    # Loop for user rating histogram
    {
      user_bins <- unique(user_hist$num_ratings)  # Get unique values for user ratings
      user_bins <- user_bins[order(user_bins)]  # Sort them in ascending order
      user_table_lines <- c()
      for (bin in user_bins) {
        num_users_in_bin <- sum(user_hist$num_ratings == bin)  # Count users in each bin
        user_table_lines <- c(user_table_lines, paste("|", sprintf("%3d", bin), "|", sprintf("%5d", num_users_in_bin), " |"))
      }
      user_table_lines
    },
    "+-----+-------+",
    
    # Part 3.e) Histogram of book ratings (how many books have been rated N times?)
    "3.e) histogram of book-ratings <table(num ratings, num books)>",
    "(how many books have been rated N times?)",
    "+-----+-------+",
    "| bin | N     |",
    "+-----+-------+",
    # Loop for book rating histogram
    {
      book_bins <- unique(book_hist$num_ratings)  # Get unique values for book ratings
      book_bins <- book_bins[order(book_bins)]  # Sort them in ascending order
      book_table_lines <- c()
      for (bin in book_bins) {
        num_books_in_bin <- sum(book_hist$num_ratings == bin)  # Count books in each bin
        book_table_lines <- c(book_table_lines, paste("|", sprintf("%3d", bin), "|", sprintf("%5d", num_books_in_bin), " |"))
      }
      book_table_lines
    },
    "+-----+-------+",
    
    # Part 3.f) Top-10 rated books
    "3.f) top-10 rated books?",
    "+--------------------+-------+",
    "| name               | N     |",
    "+--------------------+-------+",
    # Loop for top-10 books
    {
      top_books_table_lines <- c()
      for (i in 1:nrow(top_books)) {
        top_books_table_lines <- c(top_books_table_lines, 
                                   paste("|", sprintf("%-18s", top_books$Book_Title[i]), "|", sprintf("%5d", top_books$num_ratings[i]), " |"))
      }
      top_books_table_lines
    },
    "+--------------------+-------+",
    
    # Part 3.g) Top-10 active users
    "3.g) top-10 active users?",
    "+--------+-------+",
    "| user id   | N     |",
    "+--------+-------+",
    # Loop for top-10 active users
    {
      top_users_table_lines <- c()
      for (i in 1:nrow(top_users)) {
        top_users_table_lines <- c(top_users_table_lines, 
                                   paste("|", sprintf("%-6s", top_users$User_ID[i]), "|", sprintf("%5d", top_users$num_ratings[i]), " |"))
      }
      top_users_table_lines
    },
    "+--------+-------+",
    
    "-------------------- END basics.txt --------------------"
  ),
  "basics.txt"
)

# Disconnecting from the database
dbDisconnect(con)
