import re
from datetime import datetime

import pandas as pd


def currentDateString():
    return str(datetime.today().strftime("%d%m%Y"))


def readcsvInputData(
    my_data_path="../input_data/",
    my_csv_file="input.tsv",
    my_cols=["row", "id", "title", "body"],
    my_col_1="title",
    my_col_2="body",
    my_sep="\t",
    my_lineterminator="\n",
    encoding="utf-8",
):
    """Read a tab-separated input file with encoding fallback."""
    path = my_data_path + my_csv_file
    try:
        df = pd.read_csv(path, sep=my_sep, lineterminator=my_lineterminator, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=my_sep, lineterminator=my_lineterminator, encoding="latin-1")
    df.columns = my_cols
    df[my_cols[2]] = df[my_cols[2]].str.strip()
    df[my_cols[3]] = df[my_cols[3]].str.strip()
    return df


def createReviewDFFromText(my_string):
    """Create a single-row DataFrame from a review text string."""
    data = {
        "row": [1],
        "id": ["Unknown_ID"],
        "title": ["N/A"],
        "body": [my_string.strip()],
    }
    return pd.DataFrame.from_dict(data, orient="columns")


def readtsvStreamlitInputData(
    my_tsv_file,
    my_cols=["row", "id", "title", "body"],
    my_col_1="title",
    my_col_2="body",
    my_sep="\t",
    my_lineterminator="\n",
    encoding="utf-8",
):
    """Load a TSV submitted via Streamlit, with encoding fallback."""
    try:
        df = pd.read_csv(my_tsv_file, sep=my_sep, lineterminator=my_lineterminator, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(my_tsv_file, sep=my_sep, lineterminator=my_lineterminator, encoding="latin-1")
    df.columns = my_cols
    df[my_cols[2]] = df[my_cols[2]].str.strip()
    df[my_cols[3]] = df[my_cols[3]].str.strip()
    return df


def appendSuffixPrefixList(lst, len_str=20, prefix="", suffix=""):
    """Append suffix to short texts (fewer than len_str words)."""
    return [x + suffix if len(re.findall(r"\w+", x)) < len_str else x for x in lst]


def convertListTostring(my_numeric_list):
    """Convert a list of values to a list of strings."""
    return list(map(str, my_numeric_list))



