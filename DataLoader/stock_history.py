import os
import pandas as pd
import numpy as np
from errors import StockNotFoundError, InsufficientDatapointsError

DATE_HEADER = "Date"

class StockHistory:

    def __init__(self, ticker, folder_path, back_window_size=10, forward_window_size=1, max_back_offset=3, max_forward_offset=4):
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

        # RSI Computation (2 weeks)
        RSI_WINDOW = 10
        df["Close_change"] = df["Close"].diff()
        df['Gain'] = df['Close_change'].apply(lambda x: x if x > 0 else 0)
        df['Loss'] = df['Close_change'].apply(lambda x: -x if x < 0 else 0)

        df["Avg_gain"] = df["Gain"].rolling(window=RSI_WINDOW, min_periods=RSI_WINDOW).mean()
        df["Avg_loss"] = df["Loss"].rolling(window=RSI_WINDOW, min_periods=RSI_WINDOW).mean()

        # Wilder's smoothing for RSI
        for i in range(RSI_WINDOW, len(df)):
            df.loc[i, 'Avg_gain'] = (df.loc[i-1, 'Avg_gain'] * (RSI_WINDOW - 1) + df.loc[i, 'Gain']) / RSI_WINDOW
            df.loc[i, 'Avg_loss'] = (df.loc[i-1, 'Avg_loss'] * (RSI_WINDOW - 1) + df.loc[i, 'Loss']) / RSI_WINDOW

        df["2_week_RSI"] = (100 - (100 / (1 + df["Avg_gain"] / df["Avg_loss"]))) / 100
        # Add 1 to numerator and denominator to avoid divide by 0 warnings and taking the log of 0
        df["Log_Volume"] = np.log((df["Volume"] + 1) / (df["Volume"] + 1).shift(-1) )

        for header in headers:
            # Use percentage change for price attributes
            df[f"{header}_pct_change"] = df[header].pct_change()
        df = df.head(-1).reset_index(drop=True)

        # Use a rolling window of 10 trading days to compute 2-week volatility (annualised)
        df["2_week_volatility"] = df["Close_pct_change"].rolling(window=10).std() * (252 ** 0.5)
        # some entries have a volatility of 0, so we add 1e-9 just to avoid taking the log of 0
        df["Log_2_week_volatility"] = np.log(df["2_week_volatility"] + 1e-9)

        # We will also use these headers for day-by-day data
        other_headers = ["2_week_RSI", "Log_Volume", "Log_2_week_volatility"]

        # Compute lagged data for past window
        for i in range(1, back_window_size + 1):
            for header in headers:
                # clip percentage changes to +/- 10%
                lagged_data[f"D-{i} {header}"] = df[f"{header}_pct_change"].shift(-i).clip(lower=-0.1, upper=0.1)
            for header in other_headers:
                lagged_data[f"D-{i} {header}"] = df[header].shift(-i)
        
        # Compute lagged data for future window i.e. labels
        # start from D+1 rather than D+0
        for i in range(1, forward_window_size + 1):
            for header in headers:
                lagged_data[f"D+{i} {header}"] = df[f"{header}_pct_change"].shift(i)
            for header in other_headers:
                lagged_data[f"D+{i} {header}"] = df[header].shift(i)

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
