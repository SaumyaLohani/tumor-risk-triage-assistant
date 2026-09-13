Tumor Risk Triage Assistant

This project predicts the likelihood that a tumor is malignant based on diagnostic measurements, then uses an LLM to explain the prediction in plain language. It's built on the Wisconsin Breast Cancer Diagnostic dataset — a widely-used clinical ML benchmark of 569 samples with measurements like radius, texture, and concavity, labeled malignant or benign.

The core model is a logistic regression classifier trained with scikit-learn, chosen for interpretability: every prediction traces back to a clear, weighted combination of measurements rather than a black box. Features were standardized before training, and the model was evaluated on a held-out 20% test split.

Rather than optimizing for raw accuracy, the model was tuned to prioritize recall on the malignant class — reaching 95.2%. In a triage context, a false negative (missing an actual malignant case) carries far more risk than a false positive (an unnecessary follow-up), so recall was the metric that mattered most. Out of 42 malignant cases in the test set, the model missed only 2. A Claude-powered explanation layer then translates each prediction into a short, plain-language summary for a non-clinical user.
