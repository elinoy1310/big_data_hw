library(DBI)
library(RSQLite)

# Connecting to the database
con <- dbConnect(RSQLite::SQLite(), "books2.db")

#------------------------------------------------------ Deleting ratings with ISBN not existing in the books table
dbExecute(con, "
  DELETE FROM BX_Book_Ratings
  WHERE ISBN NOT IN (SELECT ISBN FROM BX_Books);
")

# Checking how many rows are left
dbGetQuery(con, "SELECT COUNT(*) FROM BX_Book_Ratings")

#-----------------------------------------------------------

#--------------------------------------------------------- Deleting books with invalid ISBN
# ---- Step 1: Check ----
# Books with invalid ISBN
invalid_books <- dbGetQuery(con, "
  SELECT ISBN
  FROM BX_Books
  WHERE ISBN = '' OR ISBN GLOB '*[^0-9]*';
")

# How many invalid books
cat('Number of books with invalid ISBN:', nrow(invalid_books), "\n")

if (nrow(invalid_books) > 0) {
  # How many ratings exist for these books
  ratings_count <- dbGetQuery(con, sprintf("
    SELECT COUNT(*) as cnt
    FROM BX_Book_Ratings
    WHERE ISBN IN ('%s')
  ", paste(invalid_books$ISBN, collapse="','")))
  
  cat('Number of ratings for invalid books:', ratings_count$cnt, "\n")
}

# ---- Step 2: Deletion ----
if (nrow(invalid_books) > 0) {
  # Deleting ratings for invalid books
  dbExecute(con, sprintf("
    DELETE FROM BX_Book_Ratings
    WHERE ISBN IN ('%s')
  ", paste(invalid_books$ISBN, collapse="','")))
  
  # Deleting the invalid books themselves
  dbExecute(con, "
    DELETE FROM BX_Books
    WHERE ISBN = '' OR ISBN GLOB '*[^0-9]*';
  ")
  
  cat("Deletion of invalid books and their ratings has been performed.\n")
}
#---------------------------------------------------------------------- Unrealistic values
# Updating age
dbExecute(con, "
    UPDATE BX_Users
    SET Age = NULL
    WHERE Age < 6 OR Age > 80;
")

# Updating publication year
dbExecute(con, "
    UPDATE BX_Books
    SET Year_Of_Publication = NULL
    WHERE Year_Of_Publication > 2025;
")


#----------------------------------------------------------------------


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------- Deleting users/books whose all ratings are 0

# Deleting ratings from users whose all ratings are 0
dbExecute(con, "
    DELETE FROM BX_Book_Ratings
    WHERE User_ID IN (
        SELECT User_ID
        FROM BX_Book_Ratings
        GROUP BY User_ID
        HAVING SUM(Book_Rating) = 0
    )
");


# Deleting ratings from books whose all ratings are 0
dbExecute(con, "
    DELETE FROM BX_Book_Ratings
    WHERE ISBN IN (
        SELECT ISBN
        FROM BX_Book_Ratings
        GROUP BY ISBN
        HAVING SUM(Book_Rating) = 0
    )
");


#------------------------------------------------------------

# For ALS algorithm
dbExecute(con, "
    DELETE FROM BX_Book_Ratings
    WHERE Book_Rating =0;
");

ratings <- dbGetQuery(con, "SELECT * FROM BX_Book_Ratings")
write.csv(ratings, "BX_Book_Ratings_clean.csv", row.names = FALSE)
books_tlb <- dbGetQuery(con, "SELECT * FROM BX_Books")
write.csv(books_tlb, "BX_Book.csv", row.names = FALSE)


# Closing the connection
dbDisconnect(con)
