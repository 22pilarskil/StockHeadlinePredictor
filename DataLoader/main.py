import csv
import pandas as pd
from utils import extract_tickers, get_company_names, sanitise_text
from stock_history import StockHistory
from errors import StockNotFoundError, InsufficientDatapointsError

DATA_FOLDER_PATH = "../Data/"
STOCK_DATA_FOLDER_PATH = DATA_FOLDER_PATH + "stock_data/"
CSV_FILE = DATA_FOLDER_PATH + "analyst_ratings_processed.csv"
USER_AGENT = None # change this!

def produce_tokenised_company_dict(csv_file) -> dict:
    tickers = extract_tickers(csv_file)
    company_name_dict = get_company_names(tickers, USER_AGENT)
    tokenised_company_dict = dict()

    for (ticker, title) in company_name_dict.items():
        tokenised_company_dict[ticker] = sanitise_text(title)

    return tokenised_company_dict

def generate_filtered_csv_file(input_file_name=CSV_FILE):
    output_file_name = input_file_name[:-4] + "_filtered.csv"
    company_name_dict = produce_tokenised_company_dict(input_file_name)
    
    invalid_tickers = set()
    total_rows = 0
    valid_headline_count = 0

    with open(input_file_name, "r") as input_file, open(output_file_name, "w", newline="") as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)

        writer.writeheader()

        for row in reader:
            total_rows += 1
            if len(row) != 4:
                continue

            title = row["title"]
            ticker = row["stock"]
            if ticker not in company_name_dict:
                invalid_tickers.add(ticker)
                continue
            
            is_valid_title = False
            for word in sanitise_text(title):
                if word in company_name_dict[ticker] or word == ticker:
                    is_valid_title = True
                    break
            
            if is_valid_title:
                writer.writerow(row)
                valid_headline_count += 1

    print("Generated new CSV file {output_file_name} containing {valid}/{initial_count} headlines".format(output_file_name=output_file_name, valid=valid_headline_count, initial_count=total_rows))
    print("Invalid tickers:")
    print(invalid_tickers)
    return output_file_name

def perform_date_check(csv_file):
    ticker_history = None

    missing = 0
    total = 0

    with open(csv_file, "r") as input_file:
        reader =  csv.DictReader(input_file)

        for row in reader:
            date = row["date"]
            ticker = row["stock"]

            try:
                if ticker_history is None or ticker_history.get_ticker() != ticker:
                    ticker_history = StockHistory(ticker, )
                    print(f"Generated history for {ticker}")
                total += 1
                if not ticker_history.check_date_in_df(date):
                    missing += 1
            except StockNotFoundError:
                continue
    
    print(f"{missing}/{total} entries without corresponding dates")
            
def combine_headlines_and_stock_prices(csv_file, back_window_size=7, forward_window_size=1, max_back_offset=3, max_forward_offset=3):
    output_file_name = csv_file[:-4] + "_combined.csv"

    ticker_history = None

    total = 0
    not_found = 0
    insufficient_data = 0
    success = 0

    CHUNK_SIZE=1000

    with open(csv_file, "r") as input_file, open(output_file_name, "w") as output_file:
        write_header = True

        for chunk in pd.read_csv(input_file, chunksize=CHUNK_SIZE, parse_dates=["date"]):
            chunk = chunk.sort_values(["stock", "date"])
            combined_pd = None

            for _, row in chunk.iterrows():
                date = row["date"]
                ticker = row["stock"]
                title = row["title"]
                total += 1

                try:
                    if ticker_history is None or ticker_history.get_ticker() != ticker:
                        ticker_history = StockHistory(ticker, folder_path=STOCK_DATA_FOLDER_PATH)
                        print(f"Generated history for {ticker}")
                    
                    new_row_pd = ticker_history.get_date_window(date)
                    new_row_pd["Title"] = title
                    new_row_pd["Ticker"] = ticker

                    if combined_pd is None:
                        combined_pd = new_row_pd
                    else:
                        combined_pd = pd.concat([combined_pd, new_row_pd], axis=0)
                    success += 1

                except StockNotFoundError:
                    not_found += 1
                    continue
                except InsufficientDatapointsError:
                    insufficient_data += 1
                    continue
            if combined_pd is None:
                continue
            combined_pd.to_csv(output_file_name, index=False, mode= "w" if write_header else "a",header=write_header) 
            write_header=False
    print(f"Saved to {output_file_name}")
    print(f"Total: {total}, Success: {success}, Not found: {not_found}, Insufficient data: {insufficient_data}")
           
# run this script to generate the CSV file for the dataset

if __name__ == "__main__":
    filtered_headlines_file = generate_filtered_csv_file(CSV_FILE)
    combine_headlines_and_stock_prices(filtered_headlines_file, back_window_size=7, forward_window_size=1)        

