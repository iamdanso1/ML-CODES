"""
ETL for the National Petroleum Authority (NPA) historical price workbook.

The NPA workbook is a human-formatted spreadsheet: a banner, a merged product
header row, a field header row, a units row, then the data. Product names only
appear once above their block of columns, so they must be forward-filled across
the header before the columns can be named.

Public entry point: ``build_panel()`` -> tidy long frame + wide/monthly views.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpyxl
import pandas as pd

# --------------------------------------------------------------------------
# Naming maps
# --------------------------------------------------------------------------

PRODUCT_CODES = {
    "PREMIUM GASOLINE": "PMS",
    "GAS OIL": "AGO",
    "KEROSINE": "KERO",
    "KEROSENE": "KERO",
    "LPG": "LPG",
    "RFO": "RFO",
    "MGO LOCAL": "MGO_LOCAL",
    "PREMIX": "PREMIX",
    "PREMIX FUEL": "PREMIX",
    "GAS OIL MINES/RIG": "AGO_MINES",
    "GASOIL MINES/RIG": "AGO_MINES",   # renamed to AGO_MINES_EXP on the export sheet
    "AVIATION FUEL (ATK)": "ATK",
    "MARINE GASOIL (FOREIGN)": "MGO_FOREIGN",
}

PRODUCT_LABELS = {
    "PMS": "Petrol (PMS)",
    "AGO": "Diesel (AGO)",
    "KERO": "Kerosene",
    "LPG": "LPG",
    "RFO": "Residual Fuel Oil",
    "MGO_LOCAL": "Marine Gasoil (local)",
    "PREMIX": "Premix",
    "AGO_MINES": "Gasoil Mines/Rig (local)",
    "AGO_MINES_EXP": "Gasoil Mines/Rig (export)",
    "ATK": "Aviation Turbine Kerosene",
    "MGO_FOREIGN": "Marine Gasoil (foreign)",
}

FIELD_CODES = {
    "EX-REF": "ex_ref",
    "EX-REF PRICE": "ex_ref",
    "TAXES": "taxes",
    "EXCISE DUTY": "taxes",
    "MARGINS": "margins",
    "TAX+MGN": "tax_margin",
    "EX-PUMP": "price",
    "EX-DEPOT": "price",
    "EX-DEPOT PRICE": "price",
}

# Products quoted per kilogram rather than per litre.
PER_KG = {"LPG"}

#: The price components the NPA publishes, in build-up order.
FIELDS = ("ex_ref", "taxes", "margins", "tax_margin", "price")

#: Components that are set by policy (gazetted under LI 2186) rather than by
#: the market, and are therefore known before a pricing window opens.
POLICY_FIELDS = ("taxes", "margins", "tax_margin")


def product_columns(columns, product: str, fields=FIELDS) -> List[str]:
    """
    Columns belonging to one product.

    Matching on an exact ``PRODUCT_field`` name rather than a string prefix
    matters here: ``AGO`` is a prefix of ``AGO_MINES``, so a prefix match would
    quietly fold diesel's mining-sector twin into diesel's own feature set.
    """
    return ["%s_%s" % (product, f) for f in fields if "%s_%s" % (product, f) in columns]


@dataclass(frozen=True)
class SheetSpec:
    """Where the header rows sit in one sheet of the NPA workbook."""

    sheet: str
    product_row: int      # 0-based row holding merged product names
    field_row: int        # 0-based row holding EX-REF / TAXES / ... labels
    first_data_row: int   # 0-based first row of observations
    era: str
    quote: str            # "GHp" (Ghana pesewas) or "USc" (US cents)


SHEET_SPECS = [
    SheetSpec("Domestic Pdts (1989-Jul '07)", 4, 5, 7, "pre_2007", "GHp"),
    SheetSpec("Domestic Pdts (Aug '07-Jun", 5, 6, 8, "deregulation_lead_up", "GHp"),
    SheetSpec("Domestic Pdts (Jul 2015 - date)", 4, 5, 7, "post_deregulation", "GHp"),
    SheetSpec("Regulated Pdts (Jul 2015- date)", 4, 5, 7, "post_deregulation", "GHp"),
    SheetSpec("Export Products", 2, 3, 5, "export", "USc"),
]

DATE_MIN = pd.Timestamp("1989-01-01")
DATE_MAX = pd.Timestamp("2030-12-31")


# --------------------------------------------------------------------------
# Low-level parsing
# --------------------------------------------------------------------------

def _ffill_header(row) -> List[Optional[str]]:
    """Merged cells arrive as one value followed by blanks -- carry it across."""
    out, current = [], None
    for cell in row:
        if cell not in (None, ""):
            current = str(cell).strip()
        out.append(current)
    return out


def _resolve_sheet(names: List[str], wanted: str) -> Optional[str]:
    """Sheet names in the workbook carry stray trailing spaces and year stamps."""
    if wanted in names:
        return wanted
    key = re.sub(r"\s+", " ", wanted).strip().lower()[:18]
    for n in names:
        if re.sub(r"\s+", " ", n).strip().lower().startswith(key):
            return n
    return None


def _parse_sheet(path: Path, spec: SheetSpec, zero_audit: Optional[list] = None) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = _resolve_sheet(wb.sheetnames, spec.sheet)
    if sheet is None:
        wb.close()
        return pd.DataFrame()

    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    if len(rows) <= spec.first_data_row:
        return pd.DataFrame()

    products = _ffill_header(rows[spec.product_row])
    fields = rows[spec.field_row]

    colmap: Dict[int, Tuple[str, str]] = {}   # column index -> (product, field)
    fx_col = None
    for j, (prod, fld) in enumerate(zip(products, fields)):
        fld_key = str(fld).strip().upper() if fld not in (None, "") else ""
        prod_key = str(prod).strip().upper() if prod else ""
        if fx_col is None and ("EXCH RATE" in prod_key or "EXCHANGE RATE" in prod_key):
            fx_col = j
        code = PRODUCT_CODES.get(prod_key)
        field = FIELD_CODES.get(fld_key)
        if code and field:
            # The export sheet quotes gasoil mines/rig in US cents while the
            # domestic sheet quotes the same name in pesewas -- keep them apart.
            if spec.quote == "USc" and code == "AGO_MINES":
                code = "AGO_MINES_EXP"
            colmap[j] = (code, field)
    if fx_col is None:                      # the export sheet labels it one row down
        for j, fld in enumerate(fields):
            if fld and "EXCHANGE RATE" in str(fld).upper():
                fx_col = j
                break

    records = []
    for row in rows[spec.first_data_row:]:
        if not row or not isinstance(row[0], (dt.datetime, dt.date)):
            continue
        date = pd.Timestamp(row[0]).normalize()
        if not (DATE_MIN <= date <= DATE_MAX):
            continue                        # guards the stray year-2700 typo
        fx = row[fx_col] if fx_col is not None and fx_col < len(row) else None
        fx = float(fx) if isinstance(fx, (int, float)) and not isinstance(fx, bool) else np.nan
        for j, (code, field) in colmap.items():
            if j >= len(row):
                continue
            val = row[j]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            # A zero EX-PUMP/EX-DEPOT price is not economically valid and in
            # the source workbook represents a malformed/missing quote. Keep
            # it out of monthly means rather than treating it as a real price.
            # Other components (e.g., taxes) may legitimately be zero.
            if field == "price" and float(val) <= 0:
                if zero_audit is not None:
                    zero_audit.append({
                        "date": date,
                        "product": code,
                        "field": field,
                        "source_value": float(val),
                        "quote": "US cents/L" if spec.quote == "USc" else "Ghana pesewas/L",
                        "action": "Excluded as invalid zero price quote",
                        "effect": "Excluded before monthly averaging; zero-valued tax/component fields are retained.",
                    })
                continue
            records.append(
                {
                    "date": date,
                    "product": code,
                    "field": field,
                    "value": float(val),
                    "fx_raw": fx,
                    "era": spec.era,
                    "quote": spec.quote,
                    "sheet": sheet,
                }
            )
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def load_raw(data_dir, workbook: Optional[str] = None) -> pd.DataFrame:
    """Parse every known sheet of the NPA workbook into one long frame."""
    data_dir = Path(data_dir)
    if workbook is None:
        candidates = sorted(data_dir.glob("Historical-Trend-of-Petroleum-Products-Prices*.xlsx"))
        if not candidates:
            raise FileNotFoundError("No NPA price workbook found in %s" % data_dir)
        # The larger file is the one carrying the longer history.
        workbook = max(candidates, key=lambda p: p.stat().st_size).name
    path = data_dir / workbook

    zero_audit = []
    frames = [_parse_sheet(path, spec, zero_audit=zero_audit) for spec in SHEET_SPECS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No recognisable sheets in %s" % path)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["date", "product", "field"], keep="first")
    raw.attrs["workbook"] = str(path)
    raw.attrs["zero_price_audit"] = zero_audit
    return raw


def _to_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pesewas / US cents to GHS or USD per litre (per kg for LPG)."""
    df = df.copy()
    df["fx_ghs_usd"] = df["fx_raw"] / 100.0          # GHp per US cent -> GHS per USD
    # The 1989-2007 sheet quotes the old cedi; 10,000 old cedis = 1 GHS.
    pre = df["era"] == "pre_2007"
    df.loc[pre, "fx_ghs_usd"] = df.loc[pre, "fx_raw"] / 10_000.0
    df.loc[pre, "value"] = df.loc[pre, "value"] / 10_000.0

    df["value_native"] = df["value"] / 100.0          # pesewas -> GHS, cents -> USD
    df["currency"] = np.where(df["quote"] == "USc", "USD", "GHS")
    df["unit"] = np.where(df["product"].isin(PER_KG), "per_kg", "per_litre")
    df["label"] = df["product"].map(PRODUCT_LABELS)
    return df


def build_panel(data_dir, workbook: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict with:
      ``long``      tidy (date, product, field, value) at NPA pricing-window frequency
      ``wide``      one column per ``PRODUCT_field``, indexed by pricing-window date
      ``monthly``   month-end resample (mean across the windows falling in the month)
      ``coverage``  per-series first/last date and observation count
    """
    raw = load_raw(data_dir, workbook)
    quality_audit = pd.DataFrame(raw.attrs.get("zero_price_audit", []),
                                 columns=["date", "product", "field", "source_value", "quote", "action", "effect"])
    tidy = _to_units(raw)

    # The pre-2007 era is excluded from the modelling panel: irregular review
    # dates and the 2007 currency redenomination break continuity. It is kept
    # in ``long`` (see the ``era`` column) for the long-run context chart.
    model_rows = tidy[tidy["era"] != "pre_2007"]

    wide = model_rows.pivot_table(
        index="date", columns=["product", "field"], values="value_native", aggfunc="mean"
    ).sort_index()
    wide.columns = ["%s_%s" % (p, f) for p, f in wide.columns]

    fx = model_rows.groupby("date")["fx_ghs_usd"].mean().sort_index().rename("fx_ghs_usd")
    wide = wide.join(fx, how="left")
    wide["fx_ghs_usd"] = wide["fx_ghs_usd"].ffill()

    monthly = wide.resample("ME").mean()
    monthly.index.name = "month"

    price_cols = [c for c in wide.columns if c.endswith("_price")]
    coverage = pd.DataFrame(
        {
            "series": price_cols,
            "product": [c.rsplit("_", 1)[0] for c in price_cols],
            "n_windows": [int(wide[c].notna().sum()) for c in price_cols],
            "first": [wide[c].first_valid_index() for c in price_cols],
            "last": [wide[c].last_valid_index() for c in price_cols],
        }
    )
    coverage["n_months"] = [int(monthly[c].notna().sum()) for c in price_cols]
    coverage["label"] = coverage["product"].map(PRODUCT_LABELS)
    ccy = model_rows.drop_duplicates("product").set_index("product")["currency"]
    unit = model_rows.drop_duplicates("product").set_index("product")["unit"]
    coverage["currency"] = coverage["product"].map(ccy)
    coverage["unit"] = coverage["product"].map(unit)
    coverage = coverage.sort_values("n_months", ascending=False).reset_index(drop=True)

    return {"long": tidy, "wide": wide, "monthly": monthly, "coverage": coverage,
            "quality_audit": quality_audit}


def save_panel(panel: Dict[str, pd.DataFrame], out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel["long"].to_csv(out_dir / "npa_long.csv", index=False)
    panel["wide"].to_csv(out_dir / "npa_windows_wide.csv")
    panel["monthly"].to_csv(out_dir / "npa_monthly_wide.csv")
    panel["coverage"].to_csv(out_dir / "series_coverage.csv", index=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    panel = build_panel(root / "data")
    save_panel(panel, root / "data" / "processed")
    print(panel["coverage"].to_string(index=False))
    print("\nmonthly frame:", panel["monthly"].shape)
    print("\ncolumns:", list(panel["monthly"].columns))
