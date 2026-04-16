from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_pipeline_artifact(
    pipeline: Any,
    output_dir: str | Path,
    artifact_name: str = "churn_pipeline.joblib",
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / artifact_name
    metadata_path = out_dir / "metadata.json"

    joblib.dump(pipeline, model_path)

    payload = {
        "artifact_name": artifact_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if metadata:
        payload.update(metadata)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_pipeline_artifact(path: str | Path):
    return joblib.load(Path(path))

