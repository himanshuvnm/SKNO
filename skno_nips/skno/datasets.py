from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr
from torch.utils.data import Dataset

PDEBENCH_PID = "doi:10.18419/DARUS-2986"
PDEBENCH_API = "https://darus.uni-stuttgart.de/api/datasets/:persistentId/"
PDEBENCH_DATAACCESS = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"
FIRMS_AREA_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{sensor}/{west},{south},{east},{north}/{days}"
GPCP_BASE = "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access"


@dataclass
class WindowConfig:
    input_steps: int = 1
    target_steps: int = 1
    stride: int = 1


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, out_path: Path, chunk_size: int = 2**20) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


class GridWindowDataset(Dataset):
    def __init__(self, frames: np.ndarray, window: WindowConfig):
        if frames.ndim != 3:
            raise ValueError(f"Expected frames [T,H,W], got {frames.shape}")
        self.frames = frames.astype(np.float32)
        self.window = window
        self.indices: List[int] = []
        total = len(self.frames)
        needed = window.input_steps + window.target_steps
        for start in range(0, total - needed + 1, window.stride):
            self.indices.append(start)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        in_end = start + self.window.input_steps
        out_end = in_end + self.window.target_steps
        x = self.frames[start:in_end].mean(axis=0, keepdims=True)
        y = self.frames[in_end:out_end].mean(axis=0, keepdims=True)
        return torch.from_numpy(x), torch.from_numpy(y)


class PDEBenchNavierStokesBuilder:
    def __init__(self, root: Path):
        self.root = _ensure_dir(root)

    def _dataset_metadata(self) -> dict:
        resp = requests.get(PDEBENCH_API, params={"persistentId": PDEBENCH_PID}, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def download(self, filename_substring: str = "Navier") -> Path:
        local_dir = _ensure_dir(self.root / "navier_stokes")
        existing = list(local_dir.glob("*.hdf5"))
        if existing:
            return existing[0]

        meta = self._dataset_metadata()
        files = meta["data"]["latestVersion"]["files"]
        candidates = []
        for item in files:
            data_file = item.get("dataFile", {})
            fname = data_file.get("filename", "")
            if filename_substring.lower() in fname.lower() and fname.lower().endswith(".hdf5"):
                size = int(data_file.get("filesize", 0))
                file_id = data_file.get("id")
                candidates.append((size, fname, file_id))
        if not candidates:
            raise RuntimeError(
                f"No PDEBench file containing '{filename_substring}' was found in dataset {PDEBENCH_PID}."
            )
        candidates.sort(key=lambda x: x[0])
        _, fname, file_id = candidates[0]
        out_path = local_dir / fname
        return _download(PDEBENCH_DATAACCESS.format(file_id=file_id), out_path)

    def build_frames(self, hdf5_path: Path, max_trajectories: int = 8, max_time: int = 16, variable_index: int = 0) -> np.ndarray:
        with h5py.File(hdf5_path, "r") as f:
            # Try common PDEBench conventions.
            arrays = []
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 4:
                    arrays.append(name)
            f.visititems(visit)
            if not arrays:
                raise RuntimeError(f"No suitable 4D/5D dataset found inside {hdf5_path}")
            data = f[arrays[0]][...]

        # Support [B,T,H,W,V] or [B,T,H,W]
        if data.ndim == 5:
            data = data[:max_trajectories, :max_time, :, :, variable_index]
        elif data.ndim == 4:
            data = data[:max_trajectories, :max_time]
        else:
            raise RuntimeError(f"Unexpected Navier-Stokes tensor shape: {data.shape}")

        b, t, h, w = data.shape
        frames = data.reshape(b * t, h, w)
        frames = np.nan_to_num(frames)
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        return frames.astype(np.float32)


class FIRMSWildfireBuilder:
    def __init__(self, root: Path):
        self.root = _ensure_dir(root)

    def download(
        self,
        map_key: str,
        bbox: Tuple[float, float, float, float],
        start_date: date,
        end_date: date,
        sensor: str = "VIIRS_SNPP_NRT",
    ) -> Path:
        days = (end_date - start_date).days + 1
        west, south, east, north = bbox
        out_dir = _ensure_dir(self.root / "wildfire")
        out_path = out_dir / f"firms_{sensor}_{start_date.isoformat()}_{end_date.isoformat()}.csv"
        if out_path.exists():
            return out_path
        url = FIRMS_AREA_API.format(
            map_key=map_key,
            sensor=sensor,
            west=west,
            south=south,
            east=east,
            north=north,
            days=days,
        )
        return _download(url, out_path)

    def build_frames(
        self,
        csv_path: Path,
        bbox: Tuple[float, float, float, float],
        start_date: date,
        end_date: date,
        grid_size: Tuple[int, int] = (64, 64),
    ) -> np.ndarray:
        west, south, east, north = bbox
        df = pd.read_csv(csv_path)
        if "acq_date" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
            raise RuntimeError("Unexpected FIRMS CSV format; expected acq_date, latitude, longitude columns.")
        df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
        df = df[(df["acq_date"] >= start_date) & (df["acq_date"] <= end_date)]

        H, W = grid_size
        frames = []
        for d in daterange(start_date, end_date):
            grid = np.zeros((H, W), dtype=np.float32)
            dd = df[df["acq_date"] == d]
            for _, row in dd.iterrows():
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                if not (west <= lon <= east and south <= lat <= north):
                    continue
                iy = min(H - 1, max(0, int((lat - south) / max(1e-8, (north - south)) * H)))
                ix = min(W - 1, max(0, int((lon - west) / max(1e-8, (east - west)) * W)))
                confidence = float(row.get("confidence", 50) if not isinstance(row.get("confidence", 50), str) else 50)
                grid[H - 1 - iy, ix] += max(1.0, confidence / 100.0)
            frames.append(grid)
        frames = np.stack(frames, axis=0)
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        return frames.astype(np.float32)


class GPCPAtmosphericBuilder:
    def __init__(self, root: Path):
        self.root = _ensure_dir(root)

    @staticmethod
    def _filename_for_day(day: date, creation_hint: str) -> str:
        return f"gpcp_v01r03_daily_d{day.strftime('%Y%m%d')}_c{creation_hint}.nc"

    def _discover_creation_tag(self, day: date) -> str:
        # Try a few plausible month-level creation dates. This keeps the downloader resilient.
        year_dir = f"{GPCP_BASE}/{day.year}/"
        index = requests.get(year_dir, timeout=120)
        index.raise_for_status()
        text = index.text
        needle = f"gpcp_v01r03_daily_d{day.strftime('%Y%m%d')}_c"
        pos = text.find(needle)
        if pos == -1:
            raise RuntimeError(f"Could not discover GPCP file for {day.isoformat()} in {year_dir}")
        suffix = text[pos + len(needle): pos + len(needle) + 8]
        return suffix

    def download(self, start_date: date, end_date: date) -> List[Path]:
        out_dir = _ensure_dir(self.root / "atmospheric")
        paths: List[Path] = []
        cache_creation: dict[int, str] = {}
        for d in daterange(start_date, end_date):
            # Creation tags can vary by processing batch; discover per day if needed.
            tag = cache_creation.get(d.toordinal())
            if tag is None:
                tag = self._discover_creation_tag(d)
                cache_creation[d.toordinal()] = tag
            fname = self._filename_for_day(d, tag)
            out_path = out_dir / fname
            if not out_path.exists():
                url = f"{GPCP_BASE}/{d.year}/{fname}"
                _download(url, out_path)
            paths.append(out_path)
        return paths

    def build_frames(
        self,
        files: Sequence[Path],
        bbox: Tuple[float, float, float, float] = (-130.0, 20.0, -60.0, 55.0),
        fallback_var_names: Sequence[str] = ("precip", "precipitation", "sat_gauge_precip", "daily_precip"),
    ) -> np.ndarray:
        west, south, east, north = bbox
        daily = []
        for f in files:
            ds = xr.open_dataset(f)
            var_name = None
            for cand in fallback_var_names:
                if cand in ds.data_vars:
                    var_name = cand
                    break
            if var_name is None:
                var_name = list(ds.data_vars.keys())[0]
            da = ds[var_name]
            lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
            lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
            if lat_name is None or lon_name is None:
                raise RuntimeError(f"Unexpected coordinate names in {f}")

            # Normalize longitudes into [-180, 180] for easier regional subsetting.
            lon_vals = da[lon_name].values
            if np.nanmax(lon_vals) > 180:
                da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
            sub = da.sel({lat_name: slice(south, north), lon_name: slice(west, east)})
            arr = np.array(sub).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim != 2:
                raise RuntimeError(f"Unexpected GPCP slice shape {arr.shape} from {f}")
            daily.append(arr)
            ds.close()
        frames = np.stack(daily, axis=0)
        frames = np.nan_to_num(frames)
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        return frames.astype(np.float32)


def build_dataset(
    dataset_name: str,
    root: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    firms_map_key: Optional[str] = None,
    navier_match: str = "Navier",
    window: Optional[WindowConfig] = None,
) -> GridWindowDataset:
    window = window or WindowConfig()
    root = Path(root)

    if dataset_name == "navier_stokes":
        builder = PDEBenchNavierStokesBuilder(root)
        path = builder.download(filename_substring=navier_match)
        frames = builder.build_frames(path)
        return GridWindowDataset(frames, window)

    if dataset_name == "wildfire":
        if bbox is None:
            bbox = (-125.0, 32.0, -113.0, 43.0)
        if start_date is None:
            start_date = "2023-08-01"
        if end_date is None:
            end_date = "2023-09-15"
        if firms_map_key is None:
            firms_map_key = os.environ.get("NASA_FIRMS_MAP_KEY")
        if not firms_map_key:
            raise RuntimeError("FIRMS MAP_KEY is required. Pass --firms-map-key or set NASA_FIRMS_MAP_KEY.")
        builder = FIRMSWildfireBuilder(root)
        s = datetime.strptime(start_date, "%Y-%m-%d").date()
        e = datetime.strptime(end_date, "%Y-%m-%d").date()
        path = builder.download(firms_map_key, bbox, s, e)
        frames = builder.build_frames(path, bbox, s, e)
        return GridWindowDataset(frames, window)

    if dataset_name == "atmospheric":
        if bbox is None:
            bbox = (-130.0, 20.0, -60.0, 55.0)
        if start_date is None:
            start_date = "2023-01-01"
        if end_date is None:
            end_date = "2023-03-31"
        builder = GPCPAtmosphericBuilder(root)
        s = datetime.strptime(start_date, "%Y-%m-%d").date()
        e = datetime.strptime(end_date, "%Y-%m-%d").date()
        files = builder.download(s, e)
        frames = builder.build_frames(files, bbox=bbox)
        return GridWindowDataset(frames, window)

    raise ValueError(f"Unknown dataset '{dataset_name}'")
