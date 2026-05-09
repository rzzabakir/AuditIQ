"""File loading and chunked reading for Excel and CSV inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".xlsx", ".csv"}
LARGE_FILE_THRESHOLD_MB = 50
CSV_CHUNK_SIZE = 100_000


def load_file(filepath: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load an .xlsx or .csv file into a DataFrame plus metadata dict."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: "
            f"{sorted(SUPPORTED_EXTENSIONS)}"
        )

    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Loading file %s (%.2f MB)", path.name, file_size_mb)

    sheet_name: Optional[str] = None
    if ext == ".xlsx":
        df, sheet_name = _load_excel(path)
    else:
        df = _load_csv(path, file_size_mb)

    metadata = {
        "filename": path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "file_size_mb": round(file_size_mb, 2),
        "sheet_name": sheet_name,
    }
    logger.info(
        "Loaded %s: %d rows x %d columns",
        path.name,
        metadata["rows"],
        metadata["columns"],
    )
    return df, metadata


def _load_excel(path: Path) -> tuple[pd.DataFrame, str]:
    """Load an Excel file; returns DataFrame and sheet name used."""
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        raise ValueError(f"Failed to open Excel file '{path.name}': {exc}") from exc

    sheet_name = xl.sheet_names[0]
    df = xl.parse(sheet_name=sheet_name)
    return df, sheet_name


def _load_csv(path: Path, file_size_mb: float) -> pd.DataFrame:
    """Load a CSV, using chunked reading for large files; handles encoding."""
    encodings = ("utf-8", "latin-1")
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            if file_size_mb > LARGE_FILE_THRESHOLD_MB:
                logger.info(
                    "Large CSV detected; using chunked reading (chunk=%d)",
                    CSV_CHUNK_SIZE,
                )
                chunks = pd.read_csv(
                    path,
                    encoding=encoding,
                    chunksize=CSV_CHUNK_SIZE,
                    low_memory=False,
                )
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(path, encoding=encoding, low_memory=False)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.warning("Encoding %s failed for %s; trying next", encoding, path.name)
            continue
        except Exception as exc:
            raise ValueError(f"Failed to read CSV '{path.name}': {exc}") from exc

    raise ValueError(
        f"Failed to decode CSV '{path.name}' with any supported encoding. "
        f"Last error: {last_error}"
    )
