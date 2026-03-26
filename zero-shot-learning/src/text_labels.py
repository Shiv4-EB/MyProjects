# Customer Review Intent Classification Labels
CUSTOMER_REVIEW_LABELS = [
    "complaint",      # Customer expressing dissatisfaction
    "inquiry",        # Customer asking questions
    "feedback",       # Customer providing positive feedback
    "suggestion",     # Customer proposing improvements
    "request",        # Customer requesting action/service
]

# Label descriptions for documentation
LABEL_DESCRIPTIONS = {
    "complaint": "Customer expressing dissatisfaction or issue",
    "inquiry": "Customer asking questions about product/service",
    "feedback": "Customer providing positive feedback or praise",
    "suggestion": "Customer proposing improvements or ideas",
    "request": "Customer requesting specific action or service",
}


def reviewClassificationLabels(my_potential_label=None):
    if my_potential_label is None:
        my_potential_label = CUSTOMER_REVIEW_LABELS
    return my_potential_label


def reviewClassificationThreshold(my_potential_threshold_value=0.59):
    return my_potential_threshold_value
