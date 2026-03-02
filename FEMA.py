#!/usr/bin/env python3
"""
Fetch data from the FEMA OpenFEMA API with pagination.

Examples
--------
# 1) Search dataset catalog for "risk index" (to find the exact entity/endpoint name)
python fetch_openfema.py --search "risk index"

# 2) Fetch ALL rows from a known dataset endpoint (entity)
python fetch_openfema.py --dataset NationalRiskIndex --out nri.jsonl

# 3) Fetch subset with OData filter + select, save to CSV
python fetch_openfema.py \
  --dataset NationalRiskIndex \
  --filter "state eq 'VA'" \
  --select "state,county,tract,riskScore" \
  --out nri_va.csv

# 4) Limit rows (useful for quick tests)
python fetch_openfema.py --dataset NationalRiskIndex --top 500 --out sample.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_API = "https://www.fema.gov/api/open"

# OpenFEMA commonly enforces max $top=1000 per request. We'll default to 1000.
# (This limit is widely referenced in community tooling and examples.)  :contentReference[oaicite:2]{index=2}
DEFAULT_PAGE_SIZE = 1000


def build_session() -> requests.Session:
    """Requests session with retries/backoff for transient errors."""
    s = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(
        {
            "User-Agent": "openfema-fetcher/1.0 (+https://www.fema.gov/about/openfema)",
            "Accept": "application/json",
        }
    )
    return s


def odata_get(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int = 60,
) -> Dict[str, Any]:
    """GET JSON with basic error handling."""
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        # Try to print useful response body
        msg = r.text[:1000]
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\n{msg}")
    return r.json()


def list_datasets(session: requests.Session, keyword: Optional[str] = None, max_rows: int = 2000) -> List[Dict[str, Any]]:
    """
    List datasets from OpenFEMA metadata endpoint.
    We fetch pages and filter client-side by keyword to be robust to field-name changes.
    """
    url = f"{BASE_API}/v1/DataSets"
    out: List[Dict[str, Any]] = []
    page_size = min(DEFAULT_PAGE_SIZE, 1000)

    skip = 0
    while len(out) < max_rows:
        params = {
            "$inlinecount": "allpages",
            "$top": page_size,
            "$skip": skip,
        }
        data = odata_get(session, url, params=params)
        rows = data.get("DataSets") or data.get("dataSets") or data.get("results") or data.get("value") or []
        if not isinstance(rows, list):
            break

        out.extend(rows)
        if len(rows) < page_size:
            break
        skip += page_size

    if keyword:
        k = keyword.lower()
        def hit(d: Dict[str, Any]) -> bool:
            blob = json.dumps(d, ensure_ascii=False).lower()
            return k in blob
        out = [d for d in out if hit(d)]

    return out[:max_rows]


def infer_rows_and_key(payload: Dict[str, Any], dataset: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    OpenFEMA responses usually have a top-level key equal to the dataset/entity name,
    e.g. {"FemaWebDisasterDeclarations": [...], "metadata": {...}}.
    We'll try a few strategies to find the list.
    """
    # Most common case: exact dataset key
    if dataset in payload and isinstance(payload[dataset], list):
        return payload[dataset], dataset

    # Case-insensitive match
    for k, v in payload.items():
        if k.lower() == dataset.lower() and isinstance(v, list):
            return v, k

    # OData 'value' fallback
    if "value" in payload and isinstance(payload["value"], list):
        return payload["value"], "value"

    # Any list field fallback
    for k, v in payload.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v, k

    return [], ""


def fetch_all(
    session: requests.Session,
    dataset: str,
    version: str = "v1",
    odata_filter: Optional[str] = None,
    odata_select: Optional[str] = None,
    orderby: Optional[str] = None,
    top: Optional[int] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sleep_s: float = 0.0,
) -> Iterable[Dict[str, Any]]:
    """
    Stream records from a dataset endpoint with pagination.
    """
    url = f"{BASE_API}/{version}/{dataset}"

    # First call: figure out total count if available
    params: Dict[str, Any] = {
        "$inlinecount": "allpages",
        "$top": min(page_size, 1000),
        "$skip": 0,
    }
    if odata_filter:
        params["$filter"] = odata_filter
    if odata_select:
        params["$select"] = odata_select
    if orderby:
        params["$orderby"] = orderby

    first = odata_get(session, url, params=params)
    rows, rows_key = infer_rows_and_key(first, dataset)
    meta = first.get("metadata") or {}
    total = meta.get("count") or meta.get("total")  # typical is metadata.count

    yielded = 0
    for r in rows:
        yield r
        yielded += 1
        if top is not None and yielded >= top:
            return

    # If we can’t see the total, just keep paging until a short page returns.
    if isinstance(total, int):
        remaining = max(total - len(rows), 0)
        pages = int(math.ceil(remaining / min(page_size, 1000)))
    else:
        pages = None

    skip = len(rows)
    page_index = 0
    while True:
        if top is not None and yielded >= top:
            return

        params["$skip"] = skip
        # adjust $top if user wants a small total
        if top is not None:
            params["$top"] = min(min(page_size, 1000), top - yielded)
        else:
            params["$top"] = min(page_size, 1000)

        data = odata_get(session, url, params=params)
        batch, _ = infer_rows_and_key(data, dataset)

        if not batch:
            return

        for r in batch:
            yield r
            yielded += 1
            if top is not None and yielded >= top:
                return

        got = len(batch)
        skip += got
        page_index += 1

        if sleep_s > 0:
            time.sleep(sleep_s)

        if pages is not None and page_index >= pages:
            return

        # Stop when the API returns fewer than page size => likely last page
        if got < params["$top"]:
            return


def write_output(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    """
    Output format chosen by extension:
      - .jsonl : one JSON object per line
      - .json  : JSON array
      - .csv   : CSV (flat fields only; nested objects become JSON strings)
    """
    import os
    import csv

    ext = os.path.splitext(out_path)[1].lower()
    recs = list(records)

    if ext == ".jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    if ext == ".json" or ext == "":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        return

    if ext == ".csv":
        # Flatten keys at top-level; stringify nested values
        def norm(v: Any) -> Any:
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return v

        # Gather all columns
        cols = sorted({k for r in recs for k in r.keys()})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in recs:
                w.writerow({k: norm(r.get(k)) for k in cols})
        return

    raise ValueError(f"Unsupported output extension: {ext} (use .jsonl, .json, or .csv)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="OpenFEMA entity name, e.g., FemaWebDisasterDeclarations")
    ap.add_argument("--version", default="v1", help="API version path, e.g., v1 or v2 (default: v1)")
    ap.add_argument("--search", help="Search dataset catalog (DataSets metadata) for a keyword")
    ap.add_argument("--filter", dest="odata_filter", help="OData $filter string, e.g., \"state eq 'VA'\"")
    ap.add_argument("--select", dest="odata_select", help="OData $select string, e.g., \"state,county,riskScore\"")
    ap.add_argument("--orderby", help="OData $orderby string, e.g., \"state asc\"")
    ap.add_argument("--top", type=int, help="Total records to fetch (for sampling/testing)")
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Records per request (max 1000)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between page requests")
    ap.add_argument("--out", help="Output path (.jsonl, .json, .csv)")

    args = ap.parse_args()
    s = build_session()

    if args.search:
        rows = list_datasets(s, keyword=args.search, max_rows=2000)
        # Print a compact summary to help pick the correct entity name
        for d in rows[:50]:
            # Different metadata fields exist across versions; print what we can find
            name = d.get("name") or d.get("Name") or d.get("dataset") or d.get("Dataset") or ""
            title = d.get("title") or d.get("Title") or d.get("description") or d.get("Description") or ""
            endpoint = d.get("apiEndpoint") or d.get("ApiEndpoint") or d.get("endpoint") or d.get("Endpoint") or ""
            print(json.dumps({"name": name, "title": title, "apiEndpoint": endpoint}, ensure_ascii=False))
        return 0

    if not args.dataset or not args.out:
        ap.error("Provide either --search KEYWORD, or --dataset ENTITY --out FILE")

    rec_iter = fetch_all(
        session=s,
        dataset=args.dataset,
        version=args.version,
        odata_filter=args.odata_filter,
        odata_select=args.odata_select,
        orderby=args.orderby,
        top=args.top,
        page_size=args.page_size,
        sleep_s=args.sleep,
    )
    write_output(rec_iter, args.out)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())