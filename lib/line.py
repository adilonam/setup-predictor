import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yfinance as yf

load_dotenv()  # Load environment variables from .env file


class Calculator:
    def __init__(self, symbol=None, period="1y", interval="1d"):
        """
        Initialize Calculator with symbol, period, and interval.

        Args:
            symbol: Stock symbol (e.g., '^GSPC', 'AAPL')
            period: Time period for data (default: "1y")
            interval: Data interval (default: "1d")
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval

        # Calculate timestamp based on interval
        # Convert interval to timedelta
        interval_map = {
            '1d': pd.Timedelta(days=1),
            '5d': pd.Timedelta(days=5),
            '1wk': pd.Timedelta(weeks=1),
            '1mo': pd.Timedelta(days=30),  # Approximate
            '3mo': pd.Timedelta(days=90),  # Approximate
        }

        # Default to 1 day if interval not found
        self.interval_timestamp = interval_map.get(
            interval, pd.Timedelta(days=1))

    def download_data(self, symbol=None, period=None, interval=None):
        """
        Download historical stock data using yfinance.
        Always downloads daily data, then resamples to requested interval.
        Uses instance attributes if parameters not provided.

        Args:
            symbol: Stock symbol (optional, uses self.symbol if not provided)
            period: Time period for data (optional, uses self.period if not provided)
            interval: Data interval (optional, uses self.interval if not provided)

        Returns:
            DataFrame with OHLC data (simple columns, not MultiIndex), or empty DataFrame if download fails
        """
        # Use instance attributes if parameters not provided
        symbol = symbol or self.symbol
        period = period or self.period
        interval = interval or self.interval

        if symbol is None:
            raise ValueError(
                "Symbol must be provided either in __init__ or as parameter")

        try:
            # Always download daily data, then resample to requested interval
            data = yf.download(symbol, period=period, interval="1d")

            # Convert MultiIndex to simple columns if needed
            if isinstance(data.columns, pd.MultiIndex):
                # If MultiIndex, extract the data for the symbol
                if symbol in data.columns.levels[1]:
                    data = data.xs(symbol, level=1, axis=1)
                else:
                    # If symbol not in MultiIndex, take first column set
                    first_symbol = data.columns.get_level_values(1).unique()[0]
                    data = data.xs(first_symbol, level=1, axis=1)

            # Resample to requested interval if not daily
            if interval != "1d":
                # Map interval to pandas resample frequency
                interval_map = {
                    '5d': '5D',
                    '1wk': 'W',
                    '1mo': 'M',
                    '3mo': 'Q',  # Quarterly
                }

                resample_freq = interval_map.get(interval)
                if resample_freq:
                    # Resample OHLC data
                    # Open: first value, High: max, Low: min, Close: last
                    resampled = pd.DataFrame()
                    resampled['Open'] = data['Open'].resample(
                        resample_freq).first()
                    resampled['High'] = data['High'].resample(
                        resample_freq).max()
                    resampled['Low'] = data['Low'].resample(
                        resample_freq).min()
                    resampled['Close'] = data['Close'].resample(
                        resample_freq).last()

                    # Add Volume if present
                    if 'Volume' in data.columns:
                        resampled['Volume'] = data['Volume'].resample(
                            resample_freq).sum()

                    # Add other columns if present (e.g., Dividends, Stock Splits)
                    for col in data.columns:
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            resampled[col] = data[col].resample(
                                resample_freq).last()

                    data = resampled.dropna()

            # Add a fake row at the end: similar to last row, Close = Open, timestamp = last_index + interval_timestamp
            if not data.empty:
                last_row = data.iloc[-1].copy()
                last_index = data.index[-1]

                # Create new row with Close = Open (using last row's Close as the new Open)
                new_row = last_row.copy()
                new_row['Open'] = last_row['Close']
                new_row['Close'] = last_row['Close']  # Close equals Open
                new_row['High'] = last_row['Close']  # High equals Close
                new_row['Low'] = last_row['Close']   # Low equals Close

                # Set Volume to 0 or keep same (using 0 for fake data)
                if 'Volume' in new_row:
                    new_row['Volume'] = 0

                # Set other columns to last row's values
                for col in new_row.index:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        new_row[col] = last_row[col]

                # Create new timestamp: last index + interval_timestamp
                new_timestamp = last_index + self.interval_timestamp

                # Append the new row
                new_row_df = pd.DataFrame([new_row], index=[new_timestamp])
                data = pd.concat([data, new_row_df])

            return data
        except Exception as e:
            # Return empty DataFrame on error
            return pd.DataFrame()

    def calculate_dots(self, df):
        # Extract High, Low, and Close (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Calculate H+L+Close for each bar
        hlc = high + low + close

        # Calculate the dots value for each date
        # Formula: [(H+L+Close Bar -2) + (H+L+Close Bar -1) + (H+L+Close Bar current)] ÷ 9
        dots_values = []
        for i in range(len(hlc)):
            if i < 2:
                # For first two bars, we don't have enough previous data, set to NaN
                dots_values.append(pd.NA)
            else:
                # Current bar (i), previous bar (i-1), previous of previous bar (i-2)
                current = hlc.iloc[i]
                prev_1 = hlc.iloc[i-1]
                prev_2 = hlc.iloc[i-2]
                # Sum and divide by 9
                dots_value = (current + prev_1 + prev_2) / 9
                dots_values.append(dots_value)

        # Create DataFrame with same index as SP500
        dots = pd.DataFrame({
            'dots': dots_values
        }, index=df.index)

        # Filter out NaN values
        dots_valid = dots[dots['dots'].notna()].copy()

        # Calculate x positions: place dots between current date and next date
        if len(dots_valid) > 0:
            dots_x_positions = []
            # Use interval_timestamp/2 to offset from current date
            timestamp_offset = self.interval_timestamp / 2

            for date in dots_valid.index:
                # Add timestamp_offset to current date
                x_position = pd.Timestamp(date) + timestamp_offset
                dots_x_positions.append(x_position)

            dots_valid['x_position'] = dots_x_positions

        return dots_valid

    def get_trend(self, df, dots, index=-1):
        """
        Determine trend based on three consecutive closes compared to their dots values.

        Args:
            df: DataFrame with OHLC data
            dots: Dots DataFrame (from calculate_dots) or Series
            index: Index to check from (default -1 for last position)

        Returns:
            "UP" if three consecutive closes are all higher than their corresponding dots
            "DOWN" if three consecutive closes are all lower than their corresponding dots
            "NULL" if neither condition is met or insufficient data
        """
        # Extract Close from dataframe (simple columns)
        close = df['Close']

        # Convert dots to Series if it's a DataFrame
        if isinstance(dots, pd.DataFrame):
            dots_series = dots['dots']
        else:
            dots_series = dots

        # Check bounds: need at least 3 bars
        if index < 0:
            pos_index = len(close) + index
        else:
            pos_index = index

        # Need at least 3 consecutive closes to check
        if pos_index < 2 or pos_index >= len(close):
            return "NULL"

        # Get the date indices for the last 3 closes (index-2, index-1, index)
        date_idx_0 = close.index[pos_index-1]
        date_idx_1 = close.index[pos_index-2]
        date_idx_2 = close.index[pos_index - 3]

        # Get the three closes
        close_0 = close.iloc[pos_index]
        close_1 = close.iloc[pos_index - 1]
        close_2 = close.iloc[pos_index - 2]

        # Check for NaN values in closes
        if pd.isna(close_0) or pd.isna(close_1) or pd.isna(close_2):
            return "NULL"

        # Get corresponding dots values using index (since dots_valid has same index as df)
        try:
            dot_0 = dots_series.loc[date_idx_0] if date_idx_0 in dots_series.index else pd.NA
            dot_1 = dots_series.loc[date_idx_1] if date_idx_1 in dots_series.index else pd.NA
            dot_2 = dots_series.loc[date_idx_2] if date_idx_2 in dots_series.index else pd.NA
        except (KeyError, IndexError):
            return "NULL"

        # Check for NaN values in dots
        if pd.isna(dot_0) or pd.isna(dot_1) or pd.isna(dot_2):
            return "NULL"

        # Check if all three closes are higher than their dots (UP trend)
        if close_0 > dot_0 and close_1 > dot_1 and close_2 > dot_2:
            return "UP"

        # Check if all three closes are lower than their dots (DOWN trend)
        if close_0 < dot_0 and close_1 < dot_1 and close_2 < dot_2:
            return "DOWN"

        # Neither condition met
        return "NULL"

    def get_5_2_resistance(self, df, index=-1):
        #  5-2 Down (resistance) lines ONLY EXIST when the High of Bar 1 is higher than the High of Bar 2. The
        # line connects the high of Bar 2 to High of Bar 1 and projects out to Bar 0.
        # Equation: [High(1) – High(2)] + High(1) = 5/2 Down for Bar 0

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        # Convert to positive index for bounds checking
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(high):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            if high.iloc[bar_1_idx] > high.iloc[bar_2_idx]:
                point_1 = (high.index[bar_2_idx], high.iloc[bar_2_idx])
                point_2 = (high.index[bar_0_idx], (high.iloc[bar_1_idx] -
                           high.iloc[bar_2_idx]) + high.iloc[bar_1_idx])
                return point_1, point_2

        return None, None

    def get_5_2_support(self, df, index=-1):
        #         5-2 Up (support) lines ONLY EXIST when the Low of Bar 1 is lower than the Low of Bar 2. The line
        # connects the Low of Bar 2 to Low of Bar 1 and projects out to Bar 0.
        # Equation: [Low(1) – Low(2)] + Low(1) = 5/2 Up for Bar 0

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(low) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(low):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Check if Low of Bar 1 is lower than Low of Bar 2
            if low.iloc[bar_1_idx] < low.iloc[bar_2_idx]:
                # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
                point_1 = (low.index[bar_2_idx], low.iloc[bar_2_idx])
                # Point 2: Projected to Bar 0 (Bar 0 date, [Low(1) - Low(2)] + Low(1))
                point_2 = (low.index[bar_0_idx], (low.iloc[bar_1_idx] -
                           low.iloc[bar_2_idx]) + low.iloc[bar_1_idx])
                return point_1, point_2

        return None, None

    def get_5_1_resistance(self, df, index=-1):
        #         5-1 Down (Resistance) line only exist when the Low of Bar 1 is HIGHER than the Low of Bar 2 AND the
        # resulting projection is ABOVE the Close of Bar 1.
        # Equation: [Low(1)-Low(2)]+Low(1) = 5-1 Down for Bar 0 (if GREATER than the close of Bar 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(low) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(low):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Check if Low of Bar 1 is HIGHER than Low of Bar 2
            if low.iloc[bar_1_idx] > low.iloc[bar_2_idx]:
                # Calculate the projection: [Low(1)-Low(2)]+Low(1)
                projection = (low.iloc[bar_1_idx] -
                              low.iloc[bar_2_idx]) + low.iloc[bar_1_idx]
                # Check if projection is GREATER than Close of Bar 1
                if projection > close.iloc[bar_1_idx]:
                    # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
                    point_1 = (low.index[bar_2_idx], low.iloc[bar_2_idx])
                    # Point 2: Projected to Bar 0 (Bar 0 date, [Low(1)-Low(2)]+Low(1))
                    point_2 = (low.index[bar_0_idx], projection)
                    return point_1, point_2

        return None, None

    def get_5_1_support(self, df, index=-1):
        #         5-1 Up (Support) line only exist when the High of Bar 1 is LOWER than the High of Bar 2 AND the
        # resulting projection is BELOW the Close of Bar 1.
        # Equation: [High(1)-High(2)]+High(1) = 5-1 Up for Bar 0 (if LESSER than the close of Bar 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(high):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Check if High of Bar 1 is LOWER than High of Bar 2
            if high.iloc[bar_1_idx] < high.iloc[bar_2_idx]:
                # Calculate the projection: [High(1)-High(2)]+High(1)
                projection = (
                    high.iloc[bar_1_idx] - high.iloc[bar_2_idx]) + high.iloc[bar_1_idx]
                # Check if projection is LESSER than Close of Bar 1
                if projection < close.iloc[bar_1_idx]:
                    # Point 1: High of Bar 2 (Bar 2 date, High(Bar 2))
                    point_1 = (high.index[bar_2_idx], high.iloc[bar_2_idx])
                    # Point 2: Projected to Bar 0 (Bar 0 date, [High(1)-High(2)]+High(1))
                    point_2 = (high.index[bar_0_idx], projection)
                    return point_1, point_2

        return None, None

    def get_5_3_support(self, df, index=-1):
        #         5-3 Up (Support) Line only exists when the Low of Bar 1 is HIGHER than the Low of Bar 2 AND the
        # resulting projection is BELOW the Close of Bar 1.
        # Equation: [Low(1)-Low(2)]+Low(1) = 5-3 Up for Bar 0 (if LESSER than the close of Bar 1)
        # (Note this is the same calculation as 5-1 Down – the difference is its relation to the close of Bar 1.)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(low) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(low):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Check if Low of Bar 1 is HIGHER than Low of Bar 2
            if low.iloc[bar_1_idx] > low.iloc[bar_2_idx]:
                # Calculate the projection: [Low(1)-Low(2)]+Low(1)
                projection = (low.iloc[bar_1_idx] -
                              low.iloc[bar_2_idx]) + low.iloc[bar_1_idx]
                # Check if projection is LESSER than Close of Bar 1
                if projection < close.iloc[bar_1_idx]:
                    # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
                    point_1 = (low.index[bar_2_idx], low.iloc[bar_2_idx])
                    # Point 2: Projected to Bar 0 (Bar 0 date, [Low(1)-Low(2)]+Low(1))
                    point_2 = (low.index[bar_0_idx], projection)
                    return point_1, point_2

        return None, None

    def get_5_3_resistance(self, df, index=-1):
        #         5-3 Down (Resistance) line only exists when the High of Bar 1 is LOWER than the High of Bar 2 AND the
        # resulting projection is ABOVE the Close of Bar 1.
        # Equation: [High(1)-High(2)]+High(1) = 5-3 Down for Bar 0 (if GREATER than the close of Bar 1)
        # (Note this is the same calculation as 5-1 UP – the difference is its relation to the close of Bar 1.)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(high):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Check if High of Bar 1 is LOWER than High of Bar 2
            if high.iloc[bar_1_idx] < high.iloc[bar_2_idx]:
                # Calculate the projection: [High(1)-High(2)]+High(1)
                projection = (
                    high.iloc[bar_1_idx] - high.iloc[bar_2_idx]) + high.iloc[bar_1_idx]
                # Check if projection is GREATER than Close of Bar 1
                if projection > close.iloc[bar_1_idx]:
                    # Point 1: High of Bar 2 (Bar 2 date, High(Bar 2))
                    point_1 = (high.index[bar_2_idx], high.iloc[bar_2_idx])
                    # Point 2: Projected to Bar 0 (Bar 0 date, [High(1)-High(2)]+High(1))
                    point_2 = (high.index[bar_0_idx], projection)
                    return point_1, point_2

        return None, None

    def get_5_9_support(self, df, index=-1):
        #         5-9 Up (Support) connects the High of Bar 2 to the Low of Bar 1 and projects to Bar 0. It is almost always
        # below the Close of Bar 1 – except in instances of upside "gap" between the High of Bar 2 and Low of Bar
        # 1.
        # Equation: Low Bar 1-[High(2)-Low(1)] = 5-9 Up for Bar 0

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(high):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Point 1: High of Bar 2 (Bar 2 date, High(Bar 2))
            point_1 = (high.index[bar_2_idx], high.iloc[bar_2_idx])
            # Point 2: Projected to Bar 0 using equation: Low(Bar 1) - [High(Bar 2) - Low(Bar 1)]
            # This simplifies to: Low(Bar 1) - High(Bar 2) + Low(Bar 1) = 2*Low(Bar 1) - High(Bar 2)
            projection = low.iloc[bar_1_idx] - \
                (high.iloc[bar_2_idx] - low.iloc[bar_1_idx])
            if projection < close.iloc[bar_1_idx]:
                point_2 = (low.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_5_9_resistance(self, df, index=-1):
        #         5-9 Down (Resistance) connects the Low of Bar 2 to the High of Bar 1 and projects to Bar 0. It is almost
        # always above the Close of Bar 1 – except in instances of downside "gap" between the Low of Bar 2 and
        # High of Bar 1.
        # Equation: High Bar 1+[High(1)-Low(2)] = 5-9 Down for Bar 0

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        if pos_index >= 2 and pos_index < len(high):
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
            point_1 = (low.index[bar_2_idx], low.iloc[bar_2_idx])
            # Point 2: Projected to Bar 0 using equation: High(Bar 1) + [High(Bar 1) - Low(Bar 2)]
            # This simplifies to: High(Bar 1) + High(Bar 1) - Low(Bar 2) = 2*High(Bar 1) - Low(Bar 2)
            projection = high.iloc[bar_1_idx] + \
                (high.iloc[bar_1_idx] - low.iloc[bar_2_idx])
            if projection > close.iloc[bar_1_idx]:
                point_2 = (high.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_6_1_support(self, df, dots, index=-1):
        #         The 6-1 Up (support) line runs from the High of Bar 1 thru Dot Bar 1 and Projects to Bar 0 BELOW the
        # Close Bar 1.
        # Equation = Dot Bar 1 – [High(1)-Dot(1)] = 6-1 Up for Bar 0 (If Lesser than Close of Bar 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        # Check if we have enough dots data (need bar_1_idx which is index-1)
        bar_1_idx = index - 1
        dots_len = len(dots) if isinstance(dots, pd.DataFrame) else len(dots)
        dots_pos_idx = dots_len + bar_1_idx if bar_1_idx < 0 else bar_1_idx

        if pos_index >= 2 and pos_index < len(high) and dots_pos_idx >= 0 and dots_pos_idx < dots_len:
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Convert dots to Series if it's a DataFrame
            if isinstance(dots, pd.DataFrame):
                dots_series = dots['dots']
            else:
                dots_series = dots

            point_1 = (high.index[bar_1_idx], high.iloc[bar_1_idx])
            projection = dots_series.iloc[bar_1_idx] - \
                (high.iloc[bar_1_idx] - dots_series.iloc[bar_1_idx])
            if projection < close.iloc[bar_1_idx]:
                point_2 = (high.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_6_1_resistance(self, df, dots, index=-1):
        #         6-1 Down (resistance) line runs from the Low of Bar 1 thru Dot Bar 1 and Projects to Bar 0 ABOVE the
        # Close Bar 1.
        # Equation = Dot Bar 1 + [Dot(1)-Low(1)] = 6-1 Down for Bar 0 (If GREATER than Close of Bar 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(low) + index
        else:
            pos_index = index

        # Check if we have enough dots data (need bar_1_idx which is index-1)
        bar_1_idx = index - 1
        dots_len = len(dots) if isinstance(dots, pd.DataFrame) else len(dots)
        dots_pos_idx = dots_len + bar_1_idx if bar_1_idx < 0 else bar_1_idx

        if pos_index >= 2 and pos_index < len(low) and dots_pos_idx >= 0 and dots_pos_idx < dots_len:
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Convert dots to Series if it's a DataFrame
            if isinstance(dots, pd.DataFrame):
                dots_series = dots['dots']
            else:
                dots_series = dots

            point_1 = (low.index[bar_1_idx], low.iloc[bar_1_idx])
            projection = dots_series.iloc[bar_1_idx] + \
                (dots_series.iloc[bar_1_idx] - low.iloc[bar_1_idx])
            if projection > close.iloc[bar_1_idx]:
                point_2 = (low.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_6_5_resistance(self, df, dots, index=-1):
        #         6-5 DOWN is the identical calculation to 6-1 Up – except it is GREATER/EQUAL to the Close of Bar 1.
        # The 6-5 Down (resistance) line runs from the High of Bar 1 thru Dot Bar 1 and Projects to Bar 0 Above
        # the Close Bar 1.
        # Equation = Dot Bar 1 – [High(1)-Dot(1)] = 6-1 Up for Bar 0 (If GREATER/EQUAL than Close of Bar 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        # Check if we have enough dots data (need bar_1_idx which is index-1)
        bar_1_idx = index - 1
        dots_len = len(dots) if isinstance(dots, pd.DataFrame) else len(dots)
        dots_pos_idx = dots_len + bar_1_idx if bar_1_idx < 0 else bar_1_idx

        if pos_index >= 2 and pos_index < len(high) and dots_pos_idx >= 0 and dots_pos_idx < dots_len:
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Convert dots to Series if it's a DataFrame
            if isinstance(dots, pd.DataFrame):
                dots_series = dots['dots']
            else:
                dots_series = dots

            point_1 = (high.index[bar_1_idx], high.iloc[bar_1_idx])
            projection = dots_series.iloc[bar_1_idx] - \
                (high.iloc[bar_1_idx] - dots_series.iloc[bar_1_idx])
            if projection >= close.iloc[bar_1_idx]:
                point_2 = (high.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_6_5_support(self, df, dots, index=-1):
        #         6-5 UP is the identical calculation to 6-1 Down – except it is LESSER/EQUAL to the Close of Bar 1.
        # 6-5 Up (support) line runs from the Low of Bar 1 thru Dot Bar 1 and Projects to Bar 0 BELOW the Close
        # Bar 1.
        # Equation = Dot Bar 1 + [Dot(1)-Low(1)] = 6-1 Down for Bar 0 (If LESS THAN/EQUAL than Close of Bar 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(low) + index
        else:
            pos_index = index

        # Check if we have enough dots data (need bar_1_idx which is index-1)
        bar_1_idx = index - 1
        dots_len = len(dots) if isinstance(dots, pd.DataFrame) else len(dots)
        dots_pos_idx = dots_len + bar_1_idx if bar_1_idx < 0 else bar_1_idx

        if pos_index >= 2 and pos_index < len(low) and dots_pos_idx >= 0 and dots_pos_idx < dots_len:
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Convert dots to Series if it's a DataFrame
            if isinstance(dots, pd.DataFrame):
                dots_series = dots['dots']
            else:
                dots_series = dots

            point_1 = (low.index[bar_1_idx], low.iloc[bar_1_idx])
            projection = dots_series.iloc[bar_1_idx] + \
                (dots_series.iloc[bar_1_idx] - low.iloc[bar_1_idx])
            if projection <= close.iloc[bar_1_idx]:
                point_2 = (low.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_6_7_support(self, df, dots, index=-1):
        #         6-7 Up runs from the Low of Bar 1 to the Dot 1 (LESS THAN Low Bar1 ) and projects to Bar 0
        # Equation = Dot 1-[(Low(1)-Dot(1)] = 6-7 Up (providing Dot 1 is BELOW Low 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(low) + index
        else:
            pos_index = index

        # Check if we have enough dots data (need bar_1_idx which is index-1)
        bar_1_idx = index - 1
        dots_len = len(dots) if isinstance(dots, pd.DataFrame) else len(dots)
        dots_pos_idx = dots_len + bar_1_idx if bar_1_idx < 0 else bar_1_idx

        if pos_index >= 2 and pos_index < len(low) and dots_pos_idx >= 0 and dots_pos_idx < dots_len:
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Convert dots to Series if it's a DataFrame
            if isinstance(dots, pd.DataFrame):
                dots_series = dots['dots']
            else:
                dots_series = dots
            
            # Check if Dot 1 is BELOW Low 1
            if dots_series.iloc[bar_1_idx] < low.iloc[bar_1_idx]:
                # Point 1: Low of Bar 1 (Bar 1 date, Low(Bar 1))
                point_1 = (low.index[bar_1_idx], low.iloc[bar_1_idx])
                # Calculate projection: Dot(Bar 1) - [Low(Bar 1) - Dot(Bar 1)]
                # This simplifies to: Dot(Bar 1) - Low(Bar 1) + Dot(Bar 1) = 2*Dot(Bar 1) - Low(Bar 1)
                projection = dots_series.iloc[bar_1_idx] - \
                    (low.iloc[bar_1_idx] - dots_series.iloc[bar_1_idx])
                point_2 = (low.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_6_7_resistance(self, df, dots, index=-1):
        #         6-7 Down runs from the High of Bar 1 to the Dot 1 (Greater than High Bar 1) and projects to Bar 0
        # Equation = Dot 1+[(Dot(1)-High(1)] = 6-7 Down (providing Dot 1 is ABOVE High 1)

        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Check bounds: need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if index < 0:
            pos_index = len(high) + index
        else:
            pos_index = index

        # Check if we have enough dots data (need bar_1_idx which is index-1)
        bar_1_idx = index - 1
        dots_len = len(dots) if isinstance(dots, pd.DataFrame) else len(dots)
        dots_pos_idx = dots_len + bar_1_idx if bar_1_idx < 0 else bar_1_idx

        if pos_index >= 2 and pos_index < len(high) and dots_pos_idx >= 0 and dots_pos_idx < dots_len:
            bar_0_idx = index
            bar_1_idx = index - 1
            bar_2_idx = index - 2

            # Convert dots to Series if it's a DataFrame
            if isinstance(dots, pd.DataFrame):
                dots_series = dots['dots']
            else:
                dots_series = dots

            # Check if Dot 1 is ABOVE High 1
            if dots_series.iloc[bar_1_idx] > high.iloc[bar_1_idx]:
                # Point 1: High of Bar 1 (Bar 1 date, High(Bar 1))
                point_1 = (high.index[bar_1_idx], high.iloc[bar_1_idx])
                # Calculate projection: Dot(Bar 1) + [Dot(Bar 1) - High(Bar 1)]
                # This simplifies to: Dot(Bar 1) + Dot(Bar 1) - High(Bar 1) = 2*Dot(Bar 1) - High(Bar 1)
                projection = dots_series.iloc[bar_1_idx] + \
                    (dots_series.iloc[bar_1_idx] - high.iloc[bar_1_idx])
                point_2 = (high.index[bar_0_idx], projection)
                return point_1, point_2

        return None, None

    def get_prompt(self, df, resistances=None, supports=None):
        # Extract columns (simple columns)
        high = df['High']
        low = df['Low']
        close = df['Close']
        open_price = df['Open']

        # Get last 10 bars
        last_10_bars = df.tail(10)

        # Format OHLC data for the last 10 bars
        bars_data = []
        for idx, row in last_10_bars.iterrows():
            bar_info = {
                'date': str(idx),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close']
            }
            bars_data.append(bar_info)

        # Format resistance lines
        resistance_lines = []
        for res in resistances:
            if res[0] is not None and res[1] is not None:
                resistance_lines.append({
                    'point1': {'date': str(res[0][0]), 'price': res[0][1]},
                    'point2': {'date': str(res[1][0]), 'price': res[1][1]}
                })

        # Format support lines
        support_lines = []
        for sup in supports:
            if sup[0] is not None and sup[1] is not None:
                support_lines.append({
                    'point1': {'date': str(sup[0][0]), 'price': sup[0][1]},
                    'point2': {'date': str(sup[1][0]), 'price': sup[1][1]}
                })

        # Current price (last close)
        current_price = close.iloc[-1]

        # Create the prompt
        prompt = f"""You are a technical analysis expert. Analyze the following stock data for {self.symbol} and provide a brief analysis report.

LAST 10 BARS (OHLC Data):
"""
        for i, bar in enumerate(bars_data, 1):
            prompt += f"Bar {i} ({bar['date']}): Open={bar['open']:.2f}, High={bar['high']:.2f}, Low={bar['low']:.2f}, Close={bar['close']:.2f}\n"

        prompt += f"\nCURRENT PRICE: {current_price:.2f}\n\n"

        if resistance_lines:
            prompt += f"RESISTANCE LEVELS ({len(resistance_lines)} lines):\n"
            for i, res in enumerate(resistance_lines, 1):
                prompt += f"Resistance {i}: From ({res['point1']['date']}, {res['point1']['price']:.2f}) to ({res['point2']['date']}, {res['point2']['price']:.2f})\n"
        else:
            prompt += "RESISTANCE LEVELS: None identified\n"

        prompt += "\n"

        if support_lines:
            prompt += f"SUPPORT LEVELS ({len(support_lines)} lines):\n"
            for i, sup in enumerate(support_lines, 1):
                prompt += f"Support {i}: From ({sup['point1']['date']}, {sup['point1']['price']:.2f}) to ({sup['point2']['date']}, {sup['point2']['price']:.2f})\n"
        else:
            prompt += "SUPPORT LEVELS: None identified\n"

        prompt += """
Based on this data, provide:
1. A brief technical analysis (2-3 sentences)
2. The probability (as a percentage) that the price will go UP in the near term
3. The probability (as a percentage) that the price will go DOWN in the near term

Format your response as:
ANALYSIS: [your analysis]
PROBABILITY UP: [percentage]%
PROBABILITY DOWN: [percentage]%
"""

        return prompt

    def get_gpt_analysis(self, df, dots, resistances=None, supports=None, api_key=None):
        """
        Get GPT analysis for the stock data using Groq API.

        Args:
            df: DataFrame with OHLC data
            dots: Dots DataFrame
            symbol: Stock symbol
            resistances: List of resistance lines as tuples ((p1_x, p1_y), (p2_x, p2_y))
            supports: List of support lines as tuples ((p1_x, p1_y), (p2_x, p2_y))
            api_key: Groq API key (if None, tries to get from environment or Streamlit secrets)

        Returns:
            Dictionary with 'analysis', 'probability_up', 'probability_down', and 'raw_response'
        """
        if api_key is None:
            return {
                'error': 'Groq API key not provided. Please set GROQ_API_KEY environment variable or pass api_key parameter.',
                'analysis': None,
                'probability_up': None,
                'probability_down': None,
                'raw_response': None
            }

        # Generate the user prompt
        user_prompt = self.get_prompt(
            df, dots, resistances, supports)

        # System prompt
        system_prompt = """You are an expert technical analyst specializing in stock market analysis. 
You analyze price charts, support and resistance levels, and provide clear, concise technical analysis.
You provide probabilities based on technical indicators, price action, and support/resistance levels.
Always format your response exactly as requested with ANALYSIS, PROBABILITY UP, and PROBABILITY DOWN sections.
The probabilities should add up to 100%."""

        try:
            # Initialize Groq client
            client = Groq(api_key=api_key)

            # Call Groq API
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",  # Using Llama 3.1 70B model via Groq
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            raw_response = response.choices[0].message.content

            # Parse the response
            analysis = None
            probability_up = None
            probability_down = None

            lines = raw_response.split('\n')
            for line in lines:
                if line.startswith('ANALYSIS:'):
                    analysis = line.replace('ANALYSIS:', '').strip()
                elif line.startswith('PROBABILITY UP:'):
                    prob_text = line.replace(
                        'PROBABILITY UP:', '').strip().replace('%', '')
                    try:
                        probability_up = float(prob_text)
                    except:
                        pass
                elif line.startswith('PROBABILITY DOWN:'):
                    prob_text = line.replace(
                        'PROBABILITY DOWN:', '').strip().replace('%', '')
                    try:
                        probability_down = float(prob_text)
                    except:
                        pass

            return {
                'analysis': analysis,
                'probability_up': probability_up,
                'probability_down': probability_down,
                'raw_response': raw_response,
                'error': None
            }

        except Exception as e:
            return {
                'error': f'Error calling Groq API: {str(e)}',
                'analysis': None,
                'probability_up': None,
                'probability_down': None,
                'raw_response': None
            }

    def create_result_df(self, df):
        """Create result_df with all resistance/support calculations and price_up target"""
        # Calculate dots first (needed for 6-x methods)
        dots_valid = self.calculate_dots(df)

        # Start with a copy of the original dataframe
        result_df = df.copy()

        # Add dots column (simple columns)
        dots_series = pd.Series(index=df.index, dtype=float)
        dots_series.loc[dots_valid.index] = dots_valid['dots'].values
        result_df['Dots'] = dots_series

        # Define all resistance and support function names
        line_functions = [
            ('5_2_resistance', self.get_5_2_resistance, False),
            ('5_2_support', self.get_5_2_support, False),
            ('5_1_resistance', self.get_5_1_resistance, False),
            ('5_1_support', self.get_5_1_support, False),
            ('5_3_resistance', self.get_5_3_resistance, False),
            ('5_3_support', self.get_5_3_support, False),
            ('5_9_resistance', self.get_5_9_resistance, False),
            ('5_9_support', self.get_5_9_support, False),
            ('6_1_resistance', self.get_6_1_resistance, True),
            ('6_1_support', self.get_6_1_support, True),
            ('6_5_resistance', self.get_6_5_resistance, True),
            ('6_5_support', self.get_6_5_support, True),
            ('6_7_resistance', self.get_6_7_resistance, True),
            ('6_7_support', self.get_6_7_support, True),
        ]

        # Initialize columns for all resistance/support types (simple columns)
        for col_name, _, _ in line_functions:
            result_df[col_name] = np.nan

        # Initialize binary target column (simple columns)
        result_df['price_up'] = np.nan

        # Get dots as Series for 6-x functions
        if isinstance(dots_valid, pd.DataFrame):
            dots_series_for_calc = dots_valid['dots']
        else:
            dots_series_for_calc = dots_valid

        # Iterate through all valid indices
        num_rows = len(df)

        for idx in range(2, num_rows):
            index = -(num_rows - idx)

            # Calculate all lines for this index
            for col_name, func, needs_dots in line_functions:
                try:
                    if needs_dots:
                        point_1, point_2 = func(
                            df, dots_series_for_calc, index)
                    else:
                        point_1, point_2 = func(df, index)

                    # Store point2 y-value if line exists (simple columns)
                    if point_1 is not None and point_2 is not None:
                        y_value = point_2[1]
                        result_df.loc[df.index[idx], col_name] = y_value
                except Exception as e:
                    pass

        # Calculate binary target: 1 if price goes up next, 0 if down (simple columns)
        close_col = result_df['Close']
        target_col = 'price_up'

        # For each row, compare current close with next close
        for idx in range(len(result_df) - 1):
            current_close = close_col.iloc[idx]
            next_close = close_col.iloc[idx + 1]

            price_up_value = 1 if next_close > current_close else 0
            result_df.iloc[idx, result_df.columns.get_loc(
                target_col)] = price_up_value

        return result_df

    def prepare_model_data(self, result_df):
        """Prepare data for model training/prediction"""
        # Extract feature columns (all except price_up)
        if isinstance(result_df.columns, pd.MultiIndex):
            feature_cols = [
                col for col in result_df.columns if col[0] != 'price_up']
            target_col = ('price_up', self.symbol)
        else:
            feature_cols = [
                col for col in result_df.columns if col != 'price_up']
            target_col = 'price_up'

        # Create feature and target dataframes
        X = result_df[feature_cols].copy()
        y = result_df[target_col].copy()

        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Handle NaN values in features
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_imputed, y, imputer

    def create_model(self, n_features):
        """Create TensorFlow model"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        tf.keras.backend.clear_session()

        model = Sequential([
            Dense(128, activation='relu', input_shape=(n_features,)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(32, activation='relu'),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model
