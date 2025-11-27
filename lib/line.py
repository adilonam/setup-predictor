import pandas as pd


class Calculator:
    def __init__(self):
        pass

    def calculate_dots(self, df, symbol):
        # Extract High, Low, and Close from SP500 (handling MultiIndex structure)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
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
            # Get all dates from the original data index
            all_dates = df.index
            dots_x_positions = []
            last_gap = None

            for i, date in enumerate(dots_valid.index):
                # Find the index of this date in the full data
                try:
                    current_idx = all_dates.get_loc(date)
                    current_date = all_dates[current_idx]

                    # If there's a next date, calculate gap and add gap/2 to current date
                    if current_idx < len(all_dates) - 1:
                        next_date = all_dates[current_idx + 1]
                        # Calculate gap between current and next date
                        gap = pd.Timestamp(next_date) - \
                            pd.Timestamp(current_date)
                        last_gap = gap  # Store for last index
                        # Add gap/2 to current date
                        x_position = pd.Timestamp(current_date) + gap / 2
                        dots_x_positions.append(x_position)
                    else:
                        # For the last index, use last date + lastgap
                        if last_gap is not None:
                            x_position = pd.Timestamp(
                                current_date) + last_gap/2
                        else:
                            # Fallback if no previous gap (shouldn't happen normally)
                            x_position = current_date
                        dots_x_positions.append(x_position)
                except (KeyError, IndexError):
                    # Fallback: use the date itself
                    dots_x_positions.append(date)

            dots_valid['x_position'] = dots_x_positions

        return dots_valid
