"""CFPB Consumer Complaint Database -> records.

Public, refreshed nightly, and already labelled: every complaint carries a
Product, an Issue and a company response. Three real decision questions off
one state, with ground truth, for the cost of a download.

Bulk CSV:  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
Data page: https://www.consumerfinance.gov/data-research/consumer-complaints/
"""
from collections import defaultdict
import pandas as pd

from .schema import Record, Question, onehot

NARRATIVE = "Consumer complaint narrative"
PRODUCT = "Product"
RESPONSE = "Company response to consumer"

RESPONSE_OPTIONS = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
    "Untimely response",
]


def load(csv_path: str, per_product: int = 3000, chunksize: int = 100_000):
    """Stratified by Product.

    CFPB is severely skewed - credit reporting alone was 80.5% of complaints
    in FY2023. Sample it as it comes and the model learns that guessing the
    majority class at high confidence is an excellent strategy, which is the
    degenerate solution the whole calibration objective exists to prevent.
    So: hard cap per product, and stream so we never hold the file in memory.
    """
    kept = defaultdict(list)
    cols = [NARRATIVE, PRODUCT, RESPONSE, "Complaint ID"]

    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize,
                             dtype=str, on_bad_lines="skip"):
        chunk = chunk[chunk[NARRATIVE].notna() & chunk[PRODUCT].notna()]
        for _, row in chunk.iterrows():
            prod = row[PRODUCT]
            if len(kept[prod]) < per_product:
                kept[prod].append(row)
        if all(len(v) >= per_product for v in kept.values()) and len(kept) > 8:
            break

    products = sorted(kept)
    for prod, rows in kept.items():
        for row in rows:
            qs = [Question(
                id="product", type="choice", options=products,
                target=onehot(len(products), products.index(prod)),
                label_source="native",
            )]
            resp = row.get(RESPONSE)
            if isinstance(resp, str) and resp in RESPONSE_OPTIONS:
                qs.append(Question(
                    id="company_response", type="choice",
                    options=RESPONSE_OPTIONS,
                    target=onehot(len(RESPONSE_OPTIONS),
                                  RESPONSE_OPTIONS.index(resp)),
                    label_source="native",
                ))
            # narratives arrive with PII redacted as runs of X
            state = " ".join(str(row[NARRATIVE]).split())
            yield Record(state_id=f"cfpb:{row['Complaint ID']}", state=state,
                         source="cfpb", questions=qs).validate()
