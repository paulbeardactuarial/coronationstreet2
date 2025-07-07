import re
import pandas as pd


def years_diff(start_series, end_series):
    """
    Calculate the difference in years between two pandas Series of datetime values.

    Parameters:
    -----------
    start_series : pd.Series
        A pandas Series containing datetime-like values representing the start dates.
    end_series : pd.Series
        A pandas Series containing datetime-like values representing the end dates.

    Returns:
    --------
    pd.Series
        A Series of floats representing the difference in years (rounded to two decimal places)
        between the corresponding elements in `end_series` and `start_series`.

    Raises:
    -------
    TypeError:
        If either `start_series` or `end_series` is not a pandas Series.
    ValueError:
        If the lengths of the input Series do not match.
    """
    # Error handling
    if not isinstance(start_series, pd.Series):
        raise TypeError("start_series must be a pandas Series.")
    if not isinstance(end_series, pd.Series):
        raise TypeError("end_series must be a pandas Series.")
    if len(start_series) != len(end_series):
        raise ValueError(
            "start_series and end_series must be of the same length.")

    # Ensure datetime format
    try:
        start_series = pd.to_datetime(start_series)
        end_series = pd.to_datetime(end_series)
    except Exception as e:
        raise ValueError(f"Error parsing datetime values: {e}")

    # Calculate difference
    time_delta = end_series - start_series
    years_diff_series = time_delta.apply(lambda x: round(x.days / 365.25, 2))

    return years_diff_series


def clean_column_names(columns):
    """
    Clean column names with consistent capitalization, spacing, and terminology.

    Rules applied:
    1. PascalCase (each word starts with capital letter)
    2. Remove spaces (words separated by capitals)
    3. Standardize "No" to "Number"

    Args:
        columns: Index or list of column names to clean

    Returns:
        List of cleaned column names
    """
    cleaned_names = []

    for col in columns:
        col_str = str(col)

        # If column has spaces, process it
        if ' ' in col_str:
            words = col_str.split()

            # Process each word
            cleaned_words = []
            for word in words:
                # Capitalize first letter, lowercase the rest
                cleaned_word = word.capitalize()

                # Replace "No" with "Number"
                if cleaned_word == "No":
                    cleaned_word = "Number"

                cleaned_words.append(cleaned_word)

            # Join without spaces (PascalCase)
            cleaned_name = "".join(cleaned_words)
        else:
            # Already in PascalCase format, just handle "No" replacement
            cleaned_name = col_str
            # Replace standalone "No" at word boundaries with "Number"
            cleaned_name = re.sub(r'\bNo\b', 'Number', cleaned_name)

        cleaned_names.append(cleaned_name)

    return cleaned_names
