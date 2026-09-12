"""Dataset registry: declare audio + labels once, the suite does the rest."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import csv
import numpy as np
from . import validate

SR = 16000


@dataclass
class WindowSet:
    """Labeled fixed-length windows: the format the detection benchmark uses."""
    name: str
    table: str                      # csv with id, label, group, start_sample, dur_s, sr
    audio_dir: str                  # <audio_dir>/<group>.wav
    label_field: str = "y"
    labeled_field: str = "labeled"
    group_field: str = "recording"
    id_field: str = "id"
    start_field: str = "start_sample"
    dur_field: str = "dur_s"
    sr_field: str = "sr"
    source_field: str | None = "source"
    source: str | None = None       # restrict to one source value
    role: str = "both"              # train | holdout | both

    def load(self):
        rows = list(csv.DictReader(open(self.table)))
        if self.source and self.source_field:
            rows = [r for r in rows if r[self.source_field] == self.source]
        rows, dropped = validate.labeled_only(rows, self.labeled_field)
        return rows, dict(n=len(rows), dropped_unlabeled=dropped,
                          note="labeled=0 means nobody listened; those rows are excluded, "
                               "never treated as negatives")


@dataclass
class RecordingSet:
    """One continuous recording with onset/offset annotations, for frame-level work."""
    name: str
    wav: str
    intervals_csv: str              # onset_sec,offset_sec  (or StartIndex,StopIndex with src_sr)
    src_sr: int | None = None       # set if intervals are in samples at another rate
    block_sec: float = 360.0        # contiguous CV block length
    role: str = "both"
    channels: str | list[int] = "mean"

    def load(self):
        import soundfile as sf
        wav, sr = sf.read(self.wav, dtype="float32", always_2d=True)
        if isinstance(self.channels, list):
            wav = wav[:, self.channels]
        rec = wav.mean(axis=1)
        if sr != SR:
            raise validate.ValidationError(f"{self.name}: {sr} Hz, expected {SR}")
        rows = list(csv.DictReader(open(self.intervals_csv)))
        cols = rows[0].keys()
        if "onset_sec" in cols:
            iv = np.array([[float(r["onset_sec"]), float(r["offset_sec"])] for r in rows])
        elif "StartIndex" in cols:
            if not self.src_sr:
                raise validate.ValidationError(f"{self.name}: StartIndex given without src_sr")
            iv = np.array([[float(r["StartIndex"]), float(r["StopIndex"])] for r in rows]) / self.src_sr
        else:
            raise validate.ValidationError(f"{self.name}: need onset_sec/offset_sec or "
                                           f"StartIndex/StopIndex, got {list(cols)}")
        merged, clean = validate.clean_intervals(iv, len(rec) / SR, self.name)
        end, truncated = validate.annotated_span(len(rec), merged, SR, self.name)
        rec = rec[:end]
        return rec, merged, dict(**clean, span_sec=end / SR, truncated_to_annotated=truncated,
                                 note=("audio truncated to the annotated span -- scoring unheard "
                                       "audio as negative is the single most expensive bug in "
                                       "this project's history") if truncated else "")


def load_registry(path):
    """YAML -> dataset objects. See config/datasets.example.yaml."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    out = {}
    for name, spec in cfg.get("datasets", {}).items():
        kind = spec.pop("kind")
        cls = {"windows": WindowSet, "recording": RecordingSet}[kind]
        out[name] = cls(name=name, **spec)
    return out, cfg.get("settings", {})
