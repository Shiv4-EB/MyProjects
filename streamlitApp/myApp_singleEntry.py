"""Streamlit front end for zero-shot customer review intent classification."""

import sys
import time
from pathlib import Path

import polars as pl
import streamlit as st

service_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(service_root)

from src import text_preprocessing as regex
from src.text_input_data import (
    appendSuffixPrefixList,
    convertListTostring,
    createReviewDFFromText,
    currentDateString,
)
from src.text_labels import (
    reviewClassificationLabels,
    reviewClassificationThreshold,
)

import eds_entry

# ensure polars prints long strings fully
pl.Config.set_fmt_str_lengths = 1000


def dict_to_polarData(data_dict):
    return pl.DataFrame(data_dict)


def polarData_to_tsv(df_polars, sep="\t"):
    return df_polars.write_csv(sep=sep).encode("utf-8", errors="replace")


def run_classification(df, limit=1):
    """Classify up to *limit* rows of *df* and return result dict."""
    df = df.iloc[:limit]
    bodies = df["body"].str.strip().tolist()
    rows = df["row"].tolist()
    ids = convertListTostring(df["id"].str.strip().tolist())
    titles = df["title"].str.strip().tolist()

    enriched = appendSuffixPrefixList(
        bodies,
        len_str=20,
        prefix="This is a sample input that I would like to share here. ",
        suffix=". this is a sample text added at the end of input paragraph. ",
    )

    cleaned = [
        regex.regularExpressionTextCleaning(regex.textNormalize(r))
        for r in enriched
    ]

    labels = reviewClassificationLabels()
    threshold = reviewClassificationThreshold()

    start = time.time()
    preds = eds_entry.main(
        sentences=cleaned,
        labels=labels,
        multi_label=True,
        verbose=True,
        threshold_value=threshold,
    )

    print("classification took", time.time() - start)

    return {
        "Machine_Learning_Label": preds,
        "Review_Body": bodies,
        "Review_Row": rows,
        "Review_ID": ids,
        "Review_Title": titles,
    }


def main():
    st.info("Zero-shot customer review intent classification")
    st.warning("Results may take a few seconds to compute.")

    text = st.text_area("Review text to analyze:", max_chars=2_000_000)
    st.write("Entered review:", text)

    if not text:
        return

    df = createReviewDFFromText(text)
    st.dataframe(df.head(1))

    if st.button("Zero-shot Classification"):
        result = run_classification(df)
        st.markdown(
            f"<p style='font-size:30px;color:blue;'>MLS labels: {result['Machine_Learning_Label']}</p>",
            unsafe_allow_html=True,
        )

        tsv = polarData_to_tsv(dict_to_polarData(result))
        st.download_button(
            "Download TSV file",
            data=tsv,
            file_name=f"{currentDateString()}_streamlit_singleEntry_text_classification_labels.tsv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
