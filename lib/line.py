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

    def get_5_2_resistance(self, df, symbol):
        #  5-2 Down (resistance) lines ONLY EXIST when the High of Bar 1 is higher than the High of Bar 2. The
        # line connects the high of Bar 2 to High of Bar 1 and projects out to Bar 0.
        # Equation: [High(1) – High(2)] + High(1) = 5/2 Down for Bar 0

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']
        if len(high) >= 3 and high.iloc[-1] > high.iloc[-3]:
            point_1 = (high.index[-3], high.iloc[-3])
            point_2 = (high.index[-1], (high.iloc[-2] -
                       high.iloc[-3]) + high.iloc[-2])
            return point_1, point_2
        else:
            return None, None

    def get_5_2_support(self, df, symbol):
        #         5-2 Up (support) lines ONLY EXIST when the Low of Bar 1 is lower than the Low of Bar 2. The line
        # connects the Low of Bar 2 to Low of Bar 1 and projects out to Bar 0.
        # Equation: [Low(1) – Low(2)] + Low(1) = 5/2 Up for Bar 0

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Check if Low of Bar 1 is lower than Low of Bar 2
        if len(low) >= 3 and low.iloc[-2] < low.iloc[-3]:
            # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
            point_1 = (low.index[-3], low.iloc[-3])
            # Point 2: Projected to Bar 0 (Bar 0 date, [Low(1) - Low(2)] + Low(1))
            point_2 = (low.index[-1], (low.iloc[-2] -
                       low.iloc[-3]) + low.iloc[-2])
            return point_1, point_2
        else:
            return None, None

    def get_5_1_resistance(self, df, symbol):
        #         5-1 Down (Resistance) line only exist when the Low of Bar 1 is HIGHER than the Low of Bar 2 AND the
        # resulting projection is ABOVE the Close of Bar 1.
        # Equation: [Low(1)-Low(2)]+Low(1) = 5-1 Down for Bar 0 (if GREATER than the close of Bar 1)

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Check if Low of Bar 1 is HIGHER than Low of Bar 2
        if len(low) >= 3 and low.iloc[-2] > low.iloc[-3]:
            # Calculate the projection: [Low(1)-Low(2)]+Low(1)
            projection = (low.iloc[-2] - low.iloc[-3]) + low.iloc[-2]
            # Check if projection is GREATER than Close of Bar 1
            if projection > close.iloc[-2]:
                # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
                point_1 = (low.index[-3], low.iloc[-3])
                # Point 2: Projected to Bar 0 (Bar 0 date, [Low(1)-Low(2)]+Low(1))
                point_2 = (low.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_5_1_support(self, df, symbol):
        #         5-1 Up (Support) line only exist when the High of Bar 1 is LOWER than the High of Bar 2 AND the
        # resulting projection is BELOW the Close of Bar 1.
        # Equation: [High(1)-High(2)]+High(1) = 5-1 Up for Bar 0 (if LESSER than the close of Bar 1)

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Check if High of Bar 1 is LOWER than High of Bar 2
        if len(high) >= 3 and high.iloc[-2] < high.iloc[-3]:
            # Calculate the projection: [High(1)-High(2)]+High(1)
            projection = (high.iloc[-2] - high.iloc[-3]) + high.iloc[-2]
            # Check if projection is LESSER than Close of Bar 1
            if projection < close.iloc[-2]:
                # Point 1: High of Bar 2 (Bar 2 date, High(Bar 2))
                point_1 = (high.index[-3], high.iloc[-3])
                # Point 2: Projected to Bar 0 (Bar 0 date, [High(1)-High(2)]+High(1))
                point_2 = (high.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_5_3_support(self, df, symbol):
        #         5-3 Up (Support) Line only exists when the Low of Bar 1 is HIGHER than the Low of Bar 2 AND the
        # resulting projection is BELOW the Close of Bar 1.
        # Equation: [Low(1)-Low(2)]+Low(1) = 5-3 Up for Bar 0 (if LESSER than the close of Bar 1)
        # (Note this is the same calculation as 5-1 Down – the difference is its relation to the close of Bar 1.)

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Check if Low of Bar 1 is HIGHER than Low of Bar 2
        if len(low) >= 3 and low.iloc[-2] > low.iloc[-3]:
            # Calculate the projection: [Low(1)-Low(2)]+Low(1)
            projection = (low.iloc[-2] - low.iloc[-3]) + low.iloc[-2]
            # Check if projection is LESSER than Close of Bar 1
            if projection < close.iloc[-2]:
                # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
                point_1 = (low.index[-3], low.iloc[-3])
                # Point 2: Projected to Bar 0 (Bar 0 date, [Low(1)-Low(2)]+Low(1))
                point_2 = (low.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_5_3_resistance(self, df, symbol):
        #         5-3 Down (Resistance) line only exists when the High of Bar 1 is LOWER than the High of Bar 2 AND the
        # resulting projection is ABOVE the Close of Bar 1.
        # Equation: [High(1)-High(2)]+High(1) = 5-3 Down for Bar 0 (if GREATER than the close of Bar 1)
        # (Note this is the same calculation as 5-1 UP – the difference is its relation to the close of Bar 1.)

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Check if High of Bar 1 is LOWER than High of Bar 2
        if len(high) >= 3 and high.iloc[-2] < high.iloc[-3]:
            # Calculate the projection: [High(1)-High(2)]+High(1)
            projection = (high.iloc[-2] - high.iloc[-3]) + high.iloc[-2]
            # Check if projection is GREATER than Close of Bar 1
            if projection > close.iloc[-2]:
                # Point 1: High of Bar 2 (Bar 2 date, High(Bar 2))
                point_1 = (high.index[-3], high.iloc[-3])
                # Point 2: Projected to Bar 0 (Bar 0 date, [High(1)-High(2)]+High(1))
                point_2 = (high.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_5_9_support(self, df, symbol):
        #         5-9 Up (Support) connects the High of Bar 2 to the Low of Bar 1 and projects to Bar 0. It is almost always
        # below the Close of Bar 1 – except in instances of upside "gap" between the High of Bar 2 and Low of Bar
        # 1.
        # Equation: Low Bar 1-[High(2)-Low(1)] = 5-9 Up for Bar 0

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if len(high) >= 3 and len(low) >= 3 and len(close) >= 3:
            # Point 1: High of Bar 2 (Bar 2 date, High(Bar 2))
            point_1 = (high.index[-3], high.iloc[-3])
            # Point 2: Projected to Bar 0 using equation: Low(Bar 1) - [High(Bar 2) - Low(Bar 1)]
            # This simplifies to: Low(Bar 1) - High(Bar 2) + Low(Bar 1) = 2*Low(Bar 1) - High(Bar 2)
            projection = low.iloc[-2] - (high.iloc[-3] - low.iloc[-2])
            if projection < close.iloc[-2]:
                point_2 = (low.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_5_9_resistance(self, df, symbol):
        #         5-9 Down (Resistance) connects the Low of Bar 2 to the High of Bar 1 and projects to Bar 0. It is almost
        # always above the Close of Bar 1 – except in instances of downside "gap" between the Low of Bar 2 and
        # High of Bar 1.
        # Equation: High Bar 1+[High(1)-Low(2)] = 5-9 Down for Bar 0

        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        # Need at least 3 bars (Bar 0, Bar 1, Bar 2)
        if len(high) >= 3 and len(low) >= 3 and len(close) >= 3:
            # Point 1: Low of Bar 2 (Bar 2 date, Low(Bar 2))
            point_1 = (low.index[-3], low.iloc[-3])
            # Point 2: Projected to Bar 0 using equation: High(Bar 1) + [High(Bar 1) - Low(Bar 2)]
            # This simplifies to: High(Bar 1) + High(Bar 1) - Low(Bar 2) = 2*High(Bar 1) - Low(Bar 2)
            projection = high.iloc[-2] + (high.iloc[-2] - low.iloc[-3])
            if projection > close.iloc[-2]:
                point_2 = (high.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_6_1_support(self, df, dots, symbol):
        #         The 6-1 Up (support) line runs from the High of Bar 1 thru Dot Bar 1 and Projects to Bar 0 BELOW the
        # Close Bar 1.
        # Equation = Dot Bar 1 – [High(1)-Dot(1)] = 6-1 Up for Bar 0 (If Lesser than Close of Bar 1)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        if len(high) >= 3 and len(dots) >= 3 and len(close) >= 3:
            point_1 = (high.index[-2], high.iloc[-2])
            projection = dots.iloc[-2] - (high.iloc[-2] - dots.iloc[-2])
            if projection < close.iloc[-2]:
                point_2 = (high.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_6_1_resistance(self, df, dots, symbol):
        #         6-1 Down (resistance) line runs from the Low of Bar 1 thru Dot Bar 1 and Projects to Bar 0 ABOVE the
        # Close Bar 1.
        # Equation = Dot Bar 1 + [Dot(1)-Low(1)] = 6-1 Down for Bar 0 (If GREATER than Close of Bar 1)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        if len(low) >= 3 and len(dots) >= 3 and len(close) >= 3:
            point_1 = (low.index[-2], low.iloc[-2])
            projection = dots.iloc[-2] + (dots.iloc[-2] - low.iloc[-2])
            if projection > close.iloc[-2]:
                point_2 = (low.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_6_5_resistance(self, df, dots, symbol):
        #         6-5 DOWN is the identical calculation to 6-1 Up – except it is GREATER/EQUAL to the Close of Bar 1.
        # The 6-5 Down (resistance) line runs from the High of Bar 1 thru Dot Bar 1 and Projects to Bar 0 Above
        # the Close Bar 1.
        # Equation = Dot Bar 1 – [High(1)-Dot(1)] = 6-1 Up for Bar 0 (If GREATER/EQUAL than Close of Bar 1)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        if len(high) >= 3 and len(dots) >= 3 and len(close) >= 3:
            point_1 = (high.index[-2], high.iloc[-2])
            projection = dots.iloc[-2] - (high.iloc[-2] - dots.iloc[-2])
            if projection >= close.iloc[-2]:
                point_2 = (high.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_6_5_support(self, df, dots, symbol):
        #         6-5 UP is the identical calculation to 6-1 Down – except it is LESSER/EQUAL to the Close of Bar 1.
        # 6-5 Up (support) line runs from the Low of Bar 1 thru Dot Bar 1 and Projects to Bar 0 BELOW the Close
        # Bar 1.
        # Equation = Dot Bar 1 + [Dot(1)-Low(1)] = 6-1 Down for Bar 0 (If LESS THAN/EQUAL than Close of Bar 1)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        if len(low) >= 3 and len(dots) >= 3 and len(close) >= 3:
            point_1 = (low.index[-2], low.iloc[-2])
            projection = dots.iloc[-2] + (dots.iloc[-2] - low.iloc[-2])
            if projection <= close.iloc[-2]:
                point_2 = (low.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_6_7_support(self, df, dots, symbol):
        #         6-7 Up runs from the Low of Bar 1 to the Dot 1 (LESS THAN Low Bar1 ) and projects to Bar 0
        # Equation = Dot 1-[(Low(1)-Dot(1)] = 6-7 Up (providing Dot 1 is BELOW Low 1)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        if len(low) >= 3 and len(dots) >= 3 and len(close) >= 3:
            # Check if Dot 1 is BELOW Low 1
            if dots.iloc[-2] < low.iloc[-2]:
                # Point 1: Low of Bar 1 (Bar 1 date, Low(Bar 1))
                point_1 = (low.index[-2], low.iloc[-2])
                # Calculate projection: Dot(Bar 1) - [Low(Bar 1) - Dot(Bar 1)]
                # This simplifies to: Dot(Bar 1) - Low(Bar 1) + Dot(Bar 1) = 2*Dot(Bar 1) - Low(Bar 1)
                projection = dots.iloc[-2] - (low.iloc[-2] - dots.iloc[-2])
                point_2 = (low.index[-1], projection)
                return point_1, point_2

        return None, None

    def get_6_7_resistance(self, df, dots, symbol):
        #         6-7 Down runs from the High of Bar 1 to the Dot 1 (Greater than High Bar 1) and projects to Bar 0
        # Equation = Dot 1+[(Dot(1)-High(1)] = 6-7 Down (providing Dot 1 is ABOVE High 1)
        if isinstance(df.columns, pd.MultiIndex):
            high = df[('High', symbol)]
            low = df[('Low', symbol)]
            close = df[('Close', symbol)]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']

        if len(high) >= 3 and len(dots) >= 3 and len(close) >= 3:
            # Check if Dot 1 is ABOVE High 1
            if dots.iloc[-2] > high.iloc[-2]:
                # Point 1: High of Bar 1 (Bar 1 date, High(Bar 1))
                point_1 = (high.index[-2], high.iloc[-2])
                # Calculate projection: Dot(Bar 1) + [Dot(Bar 1) - High(Bar 1)]
                # This simplifies to: Dot(Bar 1) + Dot(Bar 1) - High(Bar 1) = 2*Dot(Bar 1) - High(Bar 1)
                projection = dots.iloc[-2] + (dots.iloc[-2] - high.iloc[-2])
                point_2 = (high.index[-1], projection)
                return point_1, point_2

        return None, None
