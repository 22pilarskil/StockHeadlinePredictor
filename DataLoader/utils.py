import csv
import requests
import re
import os

def extract_tickers(file_name):
    tickers = set()

    with open(file_name, "r") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) == 4: # some rows have missing data
                tickers.add(row[-1]) 
    return tickers

def get_company_names(tickers: set, user_agent: str) -> dict:
    SEC_URL = "https://www.sec.gov/files/company_tickers.json"
    
    company_name_dict = dict()
    
    response = requests.get(SEC_URL, headers={
        "User-Agent": user_agent
    })
    response.raise_for_status()
    data = response.json()

    for entry in data.values():
        if entry['ticker'].upper() in tickers:
            company_name_dict[entry['ticker'].upper()] = entry['title']
    
    return company_name_dict

def sanitise_text(name) -> list:
    cleaned_title = re.sub(r"[^\w\s]", "", name)
    tokenised_title = [word.lower() for word in cleaned_title.split()]
    return tokenised_title

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 1-indexed
char_to_idx["<PAD>"] = 0  # Add padding index

def generate_ticker_encoding(ticker: str, max_len: int) -> list[int]:
    ticker_encoding = [char_to_idx[c] for c in ticker] + [0 for i in range(max_len - len(ticker))]
    return ticker_encoding

def rename_stock_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and not filename.endswith('.csv'):
            # Create the new filename with .csv extension
            ticker = filename.split("_")[0]
            new_file_path = os.path.join(folder_path, ticker + '.csv')
        
            # Rename the file
            os.rename(file_path, new_file_path)