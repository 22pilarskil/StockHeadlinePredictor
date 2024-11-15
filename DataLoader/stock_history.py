import os
import pandas as pd
import numpy as np
from errors import StockNotFoundError, InsufficientDatapointsError

DATE_HEADER = "Date"

class StockHistory:

    def __init__(self, ticker, folder_path, back_window_size=7, forward_window_size=1, max_back_offset=3, max_forward_offset=4):
        self.ticker = ticker
        self.df = self.generate_dataframe(folder_path, back_window_size=back_window_size, forward_window_size=forward_window_size)
        self.max_back_offset = max_back_offset
        self.max_forward_offset = max_forward_offset
        
    def generate_dataframe(self, folder_path, back_window_size=7, forward_window_size=1):
        file_path = folder_path + f"{self.ticker}.csv"
        if not os.path.exists(file_path):
            raise StockNotFoundError
        

        df = pd.read_csv(file_path)
        df[DATE_HEADER] = pd.to_datetime(df[DATE_HEADER], utc=True)
        df[DATE_HEADER] = df[DATE_HEADER].dt.date
        df = df.sort_values(by=DATE_HEADER, ascending=False)

        lagged_data = {}
        headers = ["Open", "High", "Low", "Close", "Volume"]

        for header in headers:
            if header == "Volume":
                # normalise the volume
                df[f"{header}_processed"] = (df[header] - df[header].mean()) / df[header].std()
            else:
                # Use log returns for price attributes
                df[f"{header}_processed"] = np.log(df[header] / df[header].shift(-1))
        df = df.head(-1).reset_index(drop=True)

        for i in range(1, back_window_size + 1):
            for header in headers:
                lagged_data[f"D-{i} {header}"] = df[f"{header}_processed"].shift(-i)
        
        # start from D+1 rather than D+0
        for i in range(1, forward_window_size + 1):
            for header in headers:
                lagged_data[f"D+{i} {header}"] = df[f"{header}_processed"].shift(i)

        lagged_df = pd.DataFrame(lagged_data)
        combined_df = pd.concat([df[DATE_HEADER], lagged_df], axis=1).dropna().reset_index(drop=True)
        return combined_df
    
    def get_date_window(self, date) -> pd.DataFrame:
        date = pd.to_datetime(date).date()
        earliest_back_end_date = date - pd.Timedelta(days=self.max_back_offset)
        latest_forward_start_date = date + pd.Timedelta(days=self.max_forward_offset)

        sliced_back_window = self.df[(self.df[DATE_HEADER] >= earliest_back_end_date) & (self.df[DATE_HEADER] <= date)]
        sliced_forward = self.df[(self.df[DATE_HEADER] > date) & (self.df[DATE_HEADER] <= latest_forward_start_date)]

        if sliced_back_window.size < 1 or sliced_forward.size < 1:
            raise InsufficientDatapointsError
        
        return sliced_back_window.head(1)
    
    def check_date_in_df(self, date) -> bool:
        date = pd.to_datetime(date).date()
        return (date in self.pd[DATE_HEADER].values)

    def get_ticker(self):
        return self.ticker
