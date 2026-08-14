
# Data Pipeline — Books to Scrape

## Project Overview

This module demonstrates a complete data-engineering pipeline:

1. Scrape book catalogue data using requests and BeautifulSoup
2. Clean and transform the scraped data using pandas
3. Convert GBP prices to INR using the required fixed project baseline
4. Load the cleaned data into a normalized SQLite database
5. Query the database using SQL
6. Read SQL results using pandas
7. Reproduce the SQL JOIN using pandas.merge()

## Data Source

The data was scraped from:

https://books.toscrape.com/

Books to Scrape is a public website designed for scraping practice.

The project scrapes books from four categories:

- Travel
- Mystery
- Historical Fiction
- Science Fiction

The final dataset contains at least 60 books across at least 3 categories.

## Fields Collected

The scraper collects:

- title
- price
- star_rating
- availability
- category

## Data Cleaning

### Price

The original price contains the GBP currency symbol.

The currency symbol is removed and the value is converted to a numeric float column named:

price_gbp

### Rating

Text ratings are converted to integers:

- One = 1
- Two = 2
- Three = 3
- Four = 4
- Five = 5

The cleaned column is named:

rating

### Availability

The availability text is converted to a Boolean column:

in_stock

Rows containing "In stock" are represented as True.

### Missing Numeric Values

Numeric parsing failures are converted to missing values and handled using median imputation.

Rows missing essential fields such as title or category are dropped because they cannot be reliably loaded into the normalized database.

## Currency Conversion

The required project-defined fixed conversion rate is:

1 GBP = 105.50 INR

This is an artificial fixed baseline required by the assignment.

No external currency API is used.

The INR price is calculated as:

price_inr = price_gbp * 105.50

## Database Design

SQLite is used as the relational database.

The database contains two normalized tables.

### categories

- category_id — INTEGER PRIMARY KEY
- category_name — TEXT UNIQUE

### books

- book_id — INTEGER PRIMARY KEY
- title — TEXT
- price_gbp — REAL
- price_inr — REAL
- rating — INTEGER
- in_stock — INTEGER
- category_id — FOREIGN KEY referencing categories(category_id)

Relationship:

categories.category_id -> books.category_id

## SQL Queries

The project includes SQL queries demonstrating:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

The SQL query strings are saved in:

sql_queries.txt

Query outputs are saved under:

query_results/

## Pandas Validation

SQL query results are read using pd.read_sql().

The JOIN result is independently reproduced using pandas.merge().

The SQL JOIN and pandas merge results are compared to verify that they are equivalent.

## Project Files

- data_pipeline.ipynb — complete executed notebook
- cleaned_books.csv — cleaned scraped dataset
- books.db — SQLite database
- sql_queries.txt — SQL query strings
- query_results/ — saved SQL and pandas query outputs
- README.md — module documentation
- requirements.txt — required Python packages

## How to Run

Install the required Python libraries:

pip install requests beautifulsoup4 pandas

Open the notebook:

data_pipeline.ipynb

Run the notebook from beginning to end.

The notebook automatically:

1. Scrapes the website
2. Cleans the data
3. Converts GBP to INR
4. Creates the SQLite database
5. Creates the normalized tables
6. Inserts the data
7. Executes SQL queries
8. Reproduces the JOIN using pandas

No API key is required.

## Design Decisions

The project uses a normalized two-table relational design so that category information is stored once in the categories table rather than being duplicated for every book.

The fixed GBP-to-INR conversion rate of 105.50 is used exactly as specified by the assignment.
