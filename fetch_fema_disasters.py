#!/usr/bin/env python3
"""
Fetch disaster-related data from FEMA OpenFEMA API.
This script fetches data from 11 disaster-related datasets and saves them as JSON or CSV.

Usage:
    python fetch_fema_disasters.py --output-format json
    python fetch_fema_disasters.py --output-format csv
    python fetch_fema_disasters.py --output-format both
    python fetch_fema_disasters.py --limit 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_API = "https://www.fema.gov/api/open"
DEFAULT_PAGE_SIZE = 1000

# Disaster-related datasets
DISASTER_DATASETS = [
    "DisasterDeclarationsSummaries",
    "DeclarationDenials",
    "FemaWebDisasterDeclarations",
    "FemaWebDisasterSummaries",
    "FemaWebDeclarationAreas",
    "IndividualAssistanceHousingRegistrantsLargeDisasters",
    "IndividualAssistanceMultipleLossFloodProperties",
    "HazardMitigationGrantProgramDisasterSummaries",
    "PublicAssistanceFundedProjectsDetails",
    "PublicAssistanceFundedProjectsSummaries",
    "PublicAssistanceApplicants",
]


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
            "User-Agent": "fema-disaster-fetcher/1.0 (+https://www.fema.gov/about/openfema)",
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
        msg = r.text[:1000]
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}\n{msg}")
    return r.json()


def fetch_dataset(
    session: requests.Session,
    dataset: str,
    limit: Optional[int] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    sleep_seconds: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Fetch all records from a FEMA dataset with pagination.
    
    Args:
        session: Requests session
        dataset: Dataset name (entity)
        limit: Maximum number of records to fetch (None = all)
        page_size: Records per page (max 1000)
        sleep_seconds: Delay between requests
        
    Returns:
        List of records
    """
    url = f"{BASE_API}/v1/{dataset}"
    records: List[Dict[str, Any]] = []
    
    skip = 0
    while True:
        if limit and len(records) >= limit:
            break
            
        # Adjust top to not exceed limit
        top = page_size
        if limit:
            remaining = limit - len(records)
            top = min(top, remaining)
        
        params = {
            "$inlinecount": "allpages",
            "$top": top,
            "$skip": skip,
        }
        
        try:
            data = odata_get(session, url, params=params)
        except RuntimeError as e:
            print(f"Error fetching {dataset} at skip={skip}: {e}", file=sys.stderr)
            break
        
        rows = data.get("value") or data.get(dataset) or []
        
        if not rows:
            break
        
        records.extend(rows)
        
        # Check if we got all records
        total = data.get("odata.count")
        if total is not None and skip + len(rows) >= total:
            break
        
        skip += len(rows)
        time.sleep(sleep_seconds)
    
    return records[:limit] if limit else records


def save_as_json(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data as JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {filepath}")


def save_as_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data as CSV."""
    if not data:
        print(f"No data to save for {filepath}")
        return
    
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for CSV export. Install with: pip install pandas", file=sys.stderr)
        return
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch disaster-related data from FEMA OpenFEMA API"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output-dir",
        default="./fema_data",
        help="Output directory (default: ./fema_data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum records per dataset (default: None = all)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Records per API request (default: {DEFAULT_PAGE_SIZE})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session = build_session()
    
    print(f"Fetching {len(DISASTER_DATASETS)} disaster-related datasets...")
    print(f"Output format: {args.output_format}")
    print(f"Output directory: {output_dir}")
    print()
    
    for i, dataset in enumerate(DISASTER_DATASETS, 1):
        print(f"[{i}/{len(DISASTER_DATASETS)}] Fetching {dataset}...")
        
        try:
            records = fetch_dataset(
                session,
                dataset,
                limit=args.limit,
                page_size=args.page_size,
                sleep_seconds=args.sleep,
            )
            
            print(f"  ✓ Got {len(records)} records")
            
            # Save as JSON
            if args.output_format in ["json", "both"]:
                json_file = output_dir / f"{dataset}.json"
                save_as_json(records, str(json_file))
            
            # Save as CSV
            if args.output_format in ["csv", "both"]:
                csv_file = output_dir / f"{dataset}.csv"
                save_as_csv(records, str(csv_file))
            
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
    
    print()
    print("✓ All datasets fetched successfully!")


if __name__ == "__main__":
    main()
