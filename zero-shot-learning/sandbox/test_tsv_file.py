"""Batch classification script — reads input.tsv and writes labelled results."""

from pathlib import Path
import time
import sys

import polars as pl

service_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(service_root)

from src.text_preprocessing import textNormalize, regularExpressionTextCleaning
from src.text_input_data import (
    readcsvInputData,
    convertListTostring,
    appendSuffixPrefixList,
    currentDateString,
)
from src.text_labels import reviewClassificationLabels, reviewClassificationThreshold

import eds_entry

pl.Config.set_fmt_str_lengths = 1000

# ---- Configuration ----
my_input_limit = 4
my_data_path_ = "../input_data/"
my_csv_file_ = "input.tsv"
my_cols_ = ["row", "id", "title", "body"]
my_sep_ = "\t"

# ---- Load data ----
test_data_dataframe = readcsvInputData(
    my_data_path=my_data_path_,
    my_csv_file=my_csv_file_,
    my_cols=my_cols_,
    my_col_1="title",
    my_col_2="body",
    my_sep=my_sep_,
    my_lineterminator="\n",
)
test_data_dataframe = test_data_dataframe.iloc[:my_input_limit]

df_review_row_list = test_data_dataframe["row"].tolist()
df_review_id_list = convertListTostring(test_data_dataframe["id"].str.strip().tolist())
df_review_title_list = test_data_dataframe["title"].str.strip().tolist()
df_review_body_list = test_data_dataframe["body"].str.strip().tolist()

test_data = appendSuffixPrefixList(
    df_review_body_list,
    len_str=20,
    prefix="This is a sample text that I would like to share here. ",
    suffix=". this is a sample text added at the end of text paragraph. ",
)

print("Number of texts:", len(test_data))

test_data_clean = [
    regularExpressionTextCleaning(textNormalize(body)) for body in test_data
]
print("Number of texts (cleaned):", len(test_data_clean))

# ---- Classify ----
my_class_labels = reviewClassificationLabels()
my_threshold_value = reviewClassificationThreshold()
my_current_date = currentDateString()

st = time.time()
print("Starting classification ...")

returned_values = eds_entry.main(
    sentences=test_data_clean,
    labels=my_class_labels,
    multi_label=True,
    verbose=True,
    threshold_value=my_threshold_value,
)

elapsed = time.time() - st
print(f"Classification took {elapsed:.2f} seconds")

# ---- Save results ----
data = {
    "Machine_Learning_Label": returned_values,
    "Review_Body": df_review_body_list,
    "Review_Row": df_review_row_list,
    "Review_ID": df_review_id_list,
    "Review_Title": df_review_title_list,
}

dataframe_polars = pl.DataFrame(data)
output_path = f"../result_dataframe/{my_current_date}_text_classification_labels.tsv"
dataframe_polars.write_csv(output_path, sep=my_sep_)
print(dataframe_polars.head(30))
