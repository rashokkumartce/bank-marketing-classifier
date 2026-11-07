"""Download and validate the official UCI Bank Marketing additional dataset."""

import argparse
from collections import Counter
import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile


SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed",
]


def download_data(output_dir):
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        additional_bytes = archive.read("bank-additional.zip")
    with zipfile.ZipFile(io.BytesIO(additional_bytes)) as archive:
        dataset_bytes = archive.read("bank-additional/bank-additional-full.csv")

    reader = csv.DictReader(io.StringIO(dataset_bytes.decode("utf-8")), delimiter=";")
    if reader.fieldnames != FEATURES + ["y"]:
        raise ValueError(f"Unexpected UCI dataset columns: {reader.fieldnames}")
    rows = list(reader)
    counts = Counter(row["y"] for row in rows)
    if len(rows) != 41188 or counts != {"no": 36548, "yes": 4640}:
        raise ValueError(f"Unexpected UCI dataset size or targets: {len(rows)}, {counts}")
    if any(
        set(row) != set(FEATURES + ["y"])
        or any(value is None or value == "" for value in row.values())
        for row in rows
    ):
        raise ValueError("Dataset contains malformed rows or empty values")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "bank_additional_full.csv"
    temporary = output_dir / "bank_additional_full.csv.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FEATURES + ["y"])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Source: {SOURCE_URL}")
    print(f"Original CSV SHA-256: {hashlib.sha256(dataset_bytes).hexdigest()}")
    print(f"Verified: {len(rows):,} rows, {len(FEATURES)} input features")
    print(f"Targets: {dict(counts)}")
    print(f"Saved: {destination.resolve()}")
    print("'unknown' categories are present; they are not empty CSV values.")
    print("Do not reuse the previous synthetic dataset or saved models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    download_data(args.output_dir)
