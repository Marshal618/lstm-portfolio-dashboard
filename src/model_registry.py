import io
import os
import json
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd

from supabase import create_client
from tensorflow.keras.models import load_model


@dataclass(frozen=True)
class PackMeta:
    pack_id: str
    symbols: Tuple[str, ...]
    lookback: int
    epochs: int
    train_days: int
    fast_mode: bool


def _sb():
    url = os.environ.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    bucket = os.environ.get("SUPABASE_BUCKET") or os.getenv("SUPABASE_BUCKET") or "model-packs"
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets.")
    return create_client(url, key), bucket


def make_pack_id(symbols: List[str], lookback: int, epochs: int, train_days: int, fast_mode: bool) -> str:
    # stable-ish pack id; short hash prevents huge names
    s = ",".join(symbols)
    base = f"sym={s}|lb={lookback}|ep={epochs}|td={train_days}|fast={int(fast_mode)}"
    import hashlib
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"pack_{h}_lb{lookback}_ep{epochs}_td{train_days}_{'fast' if fast_mode else 'full'}"


def list_packs(prefix: str = "pack_") -> List[str]:
    sb, bucket = _sb()
    # list root; pack files stored as <pack_id>.zip
    res = sb.storage.from_(bucket).list(path="")
    names = []
    for obj in res:
        name = obj.get("name", "")
        if name.startswith(prefix) and name.endswith(".zip"):
            names.append(name[:-4])
    return sorted(names)


def download_pack(pack_id: str) -> bytes:
    sb, bucket = _sb()
    path = f"{pack_id}.zip"
    data = sb.storage.from_(bucket).download(path)
    # supabase returns bytes
    return data


def upload_pack(pack_id: str, zip_bytes: bytes) -> None:
    sb, bucket = _sb()
    path = f"{pack_id}.zip"
    sb.storage.from_(bucket).upload(
        path=path,
        file=zip_bytes,
        file_options={"content-type": "application/zip", "upsert": "true"},
    )


def build_zip_from_models(
    pack_meta: PackMeta,
    models: Dict[str, object],   # keras models
    scalers: Dict[str, object],  # sklearn scalers
) -> bytes:
    """
    Writes:
      meta.json
      models/<SYM>.keras
      scalers/<SYM>.pkl
    into a zip and returns bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(pack_meta.__dict__, indent=2))

        # Keras requires filesystem for save(); use temp dir
        with tempfile.TemporaryDirectory() as td:
            mdir = os.path.join(td, "models")
            sdir = os.path.join(td, "scalers")
            os.makedirs(mdir, exist_ok=True)
            os.makedirs(sdir, exist_ok=True)

            for sym, model in models.items():
                mpath = os.path.join(mdir, f"{sym}.keras")
                model.save(mpath)
                z.write(mpath, arcname=f"models/{sym}.keras")

            for sym, scaler in scalers.items():
                spath = os.path.join(sdir, f"{sym}.pkl")
                joblib.dump(scaler, spath)
                z.write(spath, arcname=f"scalers/{sym}.pkl")

    return buf.getvalue()


def load_models_from_zip(zip_bytes: bytes) -> Tuple[PackMeta, Dict[str, object], Dict[str, object]]:
    """
    Returns (meta, models, scalers)
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
        pack_meta = PackMeta(
            pack_id=meta["pack_id"],
            symbols=tuple(meta["symbols"]),
            lookback=int(meta["lookback"]),
            epochs=int(meta["epochs"]),
            train_days=int(meta["train_days"]),
            fast_mode=bool(meta["fast_mode"]),
        )

        models, scalers = {}, {}

        with tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            models_dir = os.path.join(td, "models")
            scalers_dir = os.path.join(td, "scalers")

            if os.path.isdir(models_dir):
                for fn in os.listdir(models_dir):
                    if fn.endswith(".keras"):
                        sym = fn.replace(".keras", "")
                        models[sym] = load_model(os.path.join(models_dir, fn), compile=False)

            if os.path.isdir(scalers_dir):
                for fn in os.listdir(scalers_dir):
                    if fn.endswith(".pkl"):
                        sym = fn.replace(".pkl", "")
                        scalers[sym] = joblib.load(os.path.join(scalers_dir, fn))

    return pack_meta, models, scalers

