from pathlib import Path
import time
import json
import re
import urllib.parse
import requests
import os

import pandas as pd
from tqdm import tqdm


DATA = Path("data")
OUT_ROOT = Path("external/xenocanto")
AUDIO_DIR = OUT_ROOT / "audio"
META_DIR = OUT_ROOT / "metadata"
OUT_META = OUT_ROOT / "xenocanto_downloads.csv"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_KEEP = {"A", "B"}
MAX_SPECIES = 50
MAX_RECORDINGS_PER_SPECIES = 20
MAX_RECORDINGS_PER_RECORDIST = 4
REQUEST_SLEEP_SEC = 0.8
XC_API_KEY = os.environ["XENO_CANTO_API_KEY"]


# For first run, keep it manageable.
MIN_DURATION_SEC = 5.0


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def parse_duration_to_sec(length):
    """
    Xeno-canto length usually looks like '0:23' or '1:02'.
    """
    if pd.isna(length):
        return None
    parts = str(length).split(":")
    try:
        parts = [int(float(p)) for p in parts]
        if len(parts) == 2:
            return 60 * parts[0] + parts[1]
        if len(parts) == 3:
            return 3600 * parts[0] + 60 * parts[1] + parts[2]
    except Exception:
        return None
    return None


def choose_target_species():
    """
    Pick rare-ish BirdCLEF targets that are birds, and attach scientific/common
    names for Xeno-canto searching.
    """
    sample = pd.read_csv(DATA / "sample_submission.csv")
    target_cols = sample.columns[1:].astype(str).tolist()

    train = pd.read_csv(DATA / "train.csv")
    train["primary_label"] = train["primary_label"].astype(str)

    # Xeno-canto is bird-focused, so start with Aves only.
    train_birds = train[
        train["class_name"].astype(str).str.lower() == "aves"
    ].copy()

    counts = train_birds["primary_label"].value_counts()

    rows = []
    for sp in target_cols:
        sub = train_birds[train_birds["primary_label"] == sp]

        # Skip non-birds / missing species.
        if len(sub) == 0:
            continue

        n = int(counts.get(sp, 0))

        sci = None
        common = None

        if "scientific_name" in sub.columns and sub["scientific_name"].notna().any():
            sci = str(sub["scientific_name"].dropna().iloc[0])

        if "common_name" in sub.columns and sub["common_name"].notna().any():
            common = str(sub["common_name"].dropna().iloc[0])

        query_name = sci or common
        if query_name is None:
            continue

        rows.append(
            {
                "target_label": sp,
                "train_count": n,
                "scientific_name": sci,
                "common_name": common,
                "query_name": query_name,
            }
        )

    df = pd.DataFrame(rows).sort_values("train_count").reset_index(drop=True)

    chosen = df.head(MAX_SPECIES).copy()

    print("Chosen bird species:")
    print(
        chosen[
            ["target_label", "train_count", "scientific_name", "common_name", "query_name"]
        ].to_string(index=False)
    )

    chosen.to_csv(OUT_ROOT / "chosen_xc_species.csv", index=False)
    return chosen.to_dict("records")


def query_xc_for_species(query_name):
    records = []

    for q in ["A", "B"]:
        page = 1
        while True:
            query = f'sp:"{query_name}" q:{q}'
            url = (
                "https://xeno-canto.org/api/3/recordings?"
                + urllib.parse.urlencode({"query": query, "page": page, "key": XC_API_KEY})
            )

            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print("bad status", r.status_code, url)
                print(r.text[:300])
                break

            js = r.json()
            recs = js.get("recordings", [])
            records.extend(recs)

            num_pages = int(js.get("numPages", 1))
            if page >= num_pages:
                break

            page += 1
            time.sleep(REQUEST_SLEEP_SEC)

    by_id = {}
    for rec in records:
        by_id[str(rec.get("id"))] = rec

    return list(by_id.values())


def select_records(records):
    """
    Keep A/B quality, duration >= 5s, cap per recordist.
    """
    selected = []
    per_recordist = {}

    # Prefer A before B, then longer clips.
    def sort_key(rec):
        q = rec.get("q", "")
        q_rank = {"A": 0, "B": 1}.get(q, 9)
        dur = parse_duration_to_sec(rec.get("length")) or 0
        return (q_rank, -dur)

    for rec in sorted(records, key=sort_key):
        q = rec.get("q", "")
        if q not in QUALITY_KEEP:
            continue

        dur = parse_duration_to_sec(rec.get("length"))
        if dur is None or dur < MIN_DURATION_SEC:
            continue

        recordist = rec.get("rec", "unknown")
        per_recordist.setdefault(recordist, 0)
        if per_recordist[recordist] >= MAX_RECORDINGS_PER_RECORDIST:
            continue

        selected.append(rec)
        per_recordist[recordist] += 1

        if len(selected) >= MAX_RECORDINGS_PER_SPECIES:
            break

    return selected


def get_audio_url(rec):
    """
    Xeno-canto API usually provides 'file' as a URL-ish field.
    Sometimes it starts with //.
    """
    f = rec.get("file", "")
    if not f:
        return None
    if f.startswith("//"):
        return "https:" + f
    if f.startswith("http"):
        return f
    return "https://xeno-canto.org" + f


def download_file(url, out_path):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def main():
    species_list = choose_target_species()

    print("\nDEBUG first species item:")
    print(species_list[0])
    print("type:", type(species_list[0]))

    all_rows = []

    for item in tqdm(species_list, desc="species"):
        sp = item["target_label"]
        query_name = item["query_name"]

        print("\nQuerying", sp, "as", query_name)
        records = query_xc_for_species(query_name)
        print("records found:", len(records))

        selected = select_records(records)
        print("selected:", len(selected))

        species_dir = AUDIO_DIR / sp
        species_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(records).to_csv(META_DIR / f"{sp}_raw.csv", index=False)
        pd.DataFrame(selected).to_csv(META_DIR / f"{sp}_selected.csv", index=False)

        for rec in selected:
            xc_id = str(rec.get("id"))
            url = get_audio_url(rec)
            if not url:
                continue

            ext = Path(urllib.parse.urlparse(url).path).suffix
            if not ext:
                ext = ".mp3"

            out_path = species_dir / f"XC{xc_id}{ext}"

            if not out_path.exists():
                try:
                    print("downloading", sp, xc_id, url)
                    download_file(url, out_path)
                    time.sleep(REQUEST_SLEEP_SEC)
                except Exception as e:
                    print("FAILED", sp, xc_id, repr(e))
                    continue

            row = dict(rec)
            row["target_label"] = sp
            row["query_name"] = query_name
            row["local_audio_path"] = str(out_path)
            row["duration_sec"] = parse_duration_to_sec(rec.get("length"))
            all_rows.append(row)

        pd.DataFrame(all_rows).to_csv(OUT_META, index=False)
        print("saved running metadata:", OUT_META, "rows:", len(all_rows))

    final = pd.DataFrame(all_rows)
    final.to_csv(OUT_META, index=False)
    print("\nDone.")
    print("downloaded rows:", len(final))
    print("saved:", OUT_META)


if __name__ == "__main__":
    main()