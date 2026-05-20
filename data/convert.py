"""
Toll Road CSV Normalizer
========================
Converts the flat toll road CSV into 4 normalized CSVs
that match the PostgreSQL schema exactly.

Usage
-----
    python convert_toll_csv.py <input.csv>

Output files (written next to the input file)
---------------------------------------------
    toll_road.csv        -> toll_road     table
    measurement.csv      -> measurement   table
    vehicle_count.csv    -> vehicle_count table
    congestion.csv       -> congestion    table
    skipped_rows.csv     -> rows that failed validation (with reason)
"""

import csv
import sys
from pathlib import Path


# ── Required columns in the source CSV ───────────────────────────────────────

REQUIRED_COLS = {
    "id", "nama_tol", "lokasi", "timestamp",
    "jumlah_mobil", "jumlah_bus", "jumlah_truck",
    "jumlah_kendaraan", "jumlah_bobot_kendaraan",
    "congestion_index", "status",
}


# ── Row validation ────────────────────────────────────────────────────────────

def validate(row: dict, line_no: int) -> list[str]:
    errors = []

    for col in ("jumlah_mobil", "jumlah_bus", "jumlah_truck", "jumlah_kendaraan"):
        val = row.get(col, "").strip()
        if not val.lstrip("-").isdigit():
            errors.append(f"line {line_no}: '{col}' not an integer: {val!r}")
        elif int(val) < 0:
            errors.append(f"line {line_no}: '{col}' is negative: {val}")

    for col in ("jumlah_bobot_kendaraan", "congestion_index"):
        val = row.get(col, "").strip()
        try:
            f = float(val)
            if col == "congestion_index" and not (0.0 <= f <= 1.0):
                errors.append(f"line {line_no}: congestion_index out of [0,1]: {val}")
        except ValueError:
            errors.append(f"line {line_no}: '{col}' not numeric: {val!r}")

    for col in ("timestamp", "nama_tol", "lokasi", "status"):
        if not row.get(col, "").strip():
            errors.append(f"line {line_no}: '{col}' is empty")

    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def convert(input_path: str) -> None:
    src     = Path(input_path)
    out_dir = src.parent

    toll_road_path     = out_dir / "toll_road.csv"
    measurement_path   = out_dir / "measurement.csv"
    vehicle_count_path = out_dir / "vehicle_count.csv"
    congestion_path    = out_dir / "congestion.csv"
    skipped_path       = out_dir / "skipped_rows.csv"

    toll_roads: dict[tuple, int] = {}  # (nama_tol, lokasi) -> toll_id
    toll_id  = 0
    meas_id  = 0
    inserted = 0
    skipped  = 0

    with (
        open(src,                newline="", encoding="utf-8") as fin,
        open(toll_road_path,     "w", newline="", encoding="utf-8") as f1,
        open(measurement_path,   "w", newline="", encoding="utf-8") as f2,
        open(vehicle_count_path, "w", newline="", encoding="utf-8") as f3,
        open(congestion_path,    "w", newline="", encoding="utf-8") as f4,
        open(skipped_path,       "w", newline="", encoding="utf-8") as f5,
    ):
        reader = csv.DictReader(fin)

        missing = REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: source CSV missing columns: {missing}")

        w1 = csv.writer(f1)
        w2 = csv.writer(f2)
        w3 = csv.writer(f3)
        w4 = csv.writer(f4)
        w5 = csv.writer(f5)

        # Headers must match SQL column names exactly
        w1.writerow(["toll_id", "nama_tol", "lokasi"])
        w2.writerow(["measurement_id", "toll_id", "timestamp"])
        w3.writerow(["measurement_id", "jumlah_mobil", "jumlah_bus",
                     "jumlah_truck", "jumlah_kendaraan", "jumlah_bobot_kendaraan"])
        w4.writerow(["measurement_id", "congestion_index", "status"])
        w5.writerow(list(reader.fieldnames) + ["skip_reason"])

        for line_no, row in enumerate(reader, start=2):
            errors = validate(row, line_no)
            if errors:
                w5.writerow(list(row.values()) + ["; ".join(errors)])
                skipped += 1
                continue

            # toll_road — deduplicate by (nama_tol, lokasi)
            key = (row["nama_tol"].strip(), row["lokasi"].strip())
            if key not in toll_roads:
                toll_id += 1
                toll_roads[key] = toll_id
                w1.writerow([toll_id, key[0], key[1]])
            current_toll_id = toll_roads[key]

            # measurement
            meas_id += 1
            w2.writerow([meas_id, current_toll_id, row["timestamp"].strip()])

            # vehicle_count
            w3.writerow([
                meas_id,
                int(row["jumlah_mobil"]),
                int(row["jumlah_bus"]),
                int(row["jumlah_truck"]),
                int(row["jumlah_kendaraan"]),
                float(row["jumlah_bobot_kendaraan"]),
            ])

            # congestion
            w4.writerow([
                meas_id,
                float(row["congestion_index"]),
                row["status"].strip(),
            ])

            inserted += 1

    # Summary
    print(f"\nDone.")
    print(f"  Inserted : {inserted} rows")
    print(f"  Skipped  : {skipped} rows  {'(see skipped_rows.csv)' if skipped else ''}")
    print(f"  Unique toll roads: {len(toll_roads)}")
    print(f"\nFiles written to: {out_dir.resolve()}")
    for f in (toll_road_path, measurement_path, vehicle_count_path, congestion_path):
        print(f"  {f.name}")
    if skipped:
        print(f"  {skipped_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python convert_toll_csv.py <input.csv>")
    convert(sys.argv[1])