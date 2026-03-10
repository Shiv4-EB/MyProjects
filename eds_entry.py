import argparse
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, List, Optional, Union

from src.auto_zero_shot_classifier import zero_shot_classifier


def _preprocess(sequences: List[str]):
    """Fix None and empty string for zero-shot inputs."""
    if isinstance(sequences, list):
        return [" " if s is None or str(s).strip() == "" else s for s in sequences]
    return sequences


def _postprocess(r: Dict):
    if r.get("sequence") == " ":
        r["scores"] = [0] * len(r["scores"])
        r["sequence"] = None
    else:
        r["scores"] = [
            round(scr, 3) if label is None or str(label).strip() != "" else 0
            for scr, label in zip(r["scores"], r["labels"])
        ]
    return r


@dataclass
class ArgSanitizer:
    sentences: str
    labels: Union[List[str], str]
    multi_label: Optional[bool] = False
    hypothesis_template: Optional[str] = None
    verbose: Optional[bool] = False

    def __post_init__(self):
        sentences_ = self.sentences
        sentences_ = [sentences_] if isinstance(sentences_, str) else sentences_
        self.sentences = sentences_
        if not self.is_list_of_string(sentences_):
            raise TypeError("sentences type must be a string or a list of string.")

        self.labels = self.convert_label_str2list(self.labels)
        labels_ = self.labels
        if len(labels_) == 0:
            raise ValueError("You must include at least one label value.")
        if not self.is_list_of_non_empty_string(labels_):
            raise TypeError(
                "labels type must be a list of non empty string or a string of comma-separated labels."
            )

        hypothesis_ = self.hypothesis_template
        if hypothesis_ is not None:
            if not isinstance(hypothesis_, str):
                raise TypeError(
                    "hypothesis_template type must be a string and contains the `{}` label placeholder"
                )
            elif hypothesis_.format("_UNK_") == hypothesis_:
                raise ValueError("The hypothesis_template must include a `{}` for label placeholder")

        if not isinstance(self.multi_label, bool):
            raise TypeError("multi_label type must be a boolean `True or False`")

        if not isinstance(self.verbose, bool):
            raise TypeError("verbose type must be a boolean `True or False`")

    def convert_label_str2list(self, labels):
        """Convert comma-separated labels string to list."""
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def is_list_of_string(self, sequences: List[str]) -> bool:
        if isinstance(sequences, list):
            for s in sequences:
                if not isinstance(s, str):
                    return False
            return True
        return False

    def is_list_of_non_empty_string(self, sequences: List[str]) -> bool:
        if isinstance(sequences, list):
            for s in sequences:
                if self.is_empty_string(s):
                    return False
                if not isinstance(s, str):
                    return False
            return True
        return False

    def is_empty_string(self, s):
        if isinstance(s, str):
            s = s.strip()
            return len(s) == 0
        return False


def main(
    sentences="",
    labels="",
    multi_label=False,
    hypothesis_template=None,
    verbose=False,
    threshold_value=0.85,
):
    """Entry point to service logic.

    `threshold_value` is kept for compatibility with existing callers.
    """
    _ = threshold_value
    my_final_class_list = []

    sentences = _preprocess(sentences)
    argfn = ArgSanitizer(
        sentences=sentences,
        labels=labels,
        multi_label=multi_label,
        hypothesis_template=hypothesis_template,
        verbose=verbose,
    )

    hypothesis_template = argfn.hypothesis_template

    results = zero_shot_classifier(
        argfn.sentences,
        candidate_labels=argfn.labels,
        hypothesis_template="The customer intent in this review is {}" if hypothesis_template is None else hypothesis_template,
        multi_label=argfn.multi_label,
    )

    for i, r in enumerate(results):
        r = _postprocess(r)

        r_sequence = r["sequence"]
        r_scores = r["scores"]
        r_labels = r["labels"]

        if verbose:
            print(f"--- example {i} ---")
            print("sequence:", r_sequence)
            print("labels:", r_labels)
            print("scores:", r_scores)

        if r_scores and r_labels:
            max_score = max(r_scores)
            top_labels = [
                r_labels[j] for j in range(len(r_scores)) if r_scores[j] == max_score
            ]
            my_final_class = ", ".join(
                f"{label} ({max_score:.3f})" for label in top_labels
            )
        else:
            my_final_class = None

        my_final_class_list.append(my_final_class)

        if verbose:
            print("\n\n")
            pprint(r, indent=3, width=150, sort_dicts=False)

    return my_final_class_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple Interface of Zero Shot deep-learning leveraging NLI models"
    )
    parser.add_argument(
        "-s",
        "--sentences",
        type=str,
        required=True,
        help="sentences (`str` or `List[str]`): The sequence(s) to classify, will be truncated if the model input is too large.",
    )

    parser.add_argument(
        "-l",
        "--labels",
        action="store",
        type=str,
        required=True,
        help="The set of possible class labels to classify each sequence into. Can be a single label, a string of comma-separated labels",
    )

    parser.add_argument(
        "-t",
        "--hypothesis_template",
        action="store",
        help="""(`str`, *optional*, defaults to `"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the
                model like `"<cls> sequence to classify <sep> This example is sports . <sep>"`.""",
    )

    parser.add_argument(
        "-m",
        "--multi_label",
        action="store_true",
        help="""(`bool`, *optional*, defaults to `False`):
                Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
                the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.""",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="print the output")

    args = parser.parse_args()
    main(
        sentences=args.sentences,
        labels=args.labels,
        multi_label=args.multi_label,
        hypothesis_template=args.hypothesis_template,
        verbose=args.verbose,
        threshold_value=0.85,
    )
