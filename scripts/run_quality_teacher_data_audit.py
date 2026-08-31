#!/usr/bin/env python3
"""Prepare and audit development-only AMP-QUALITY-TEACHER-v1 waveforms."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from fssr_nam.campaign.amp_quality_teacher_registry import (
    append_gate_event,
    gate_decisions,
)
from fssr_nam.campaign.amp_quality_teacher_v1 import (
    CAMPAIGN_VERSION,
    validate_stage_authorization,
)
from fssr_nam.campaign.quality_aa_provenance import replace_json, write_new_json
from fssr_nam.data.quality_teacher import (
    PairDescriptor,
    assigned_split,
    audit_pair_arrays,
    load_data_contract,
    prepare_authorized_pair,
    validate_source_isolation,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / ".codex_campaign/amp_quality_teacher_v1"
GATE_LEDGER = CAMPAIGN_DIR / "GATE_LEDGER.jsonl"
EVIDENCE_PATH = CAMPAIGN_DIR / "DATA_AUDIT.json"
INVALID_PATH = CAMPAIGN_DIR / "DATA_AUDIT_INVALID.json"
MANIFEST_PATH = ROOT / "datasets/manifests/amp_quality_teacher_v1_development.json"
SPLIT_PATH = ROOT / "datasets/splits/amp_quality_teacher_v1_development.json"
OUTPUT_ROOT = ROOT / "datasets/raw/amp_quality_teacher_v1"


@dataclass(frozen=True)
class ArchivePair:
    descriptor: PairDescriptor
    input_archive: str
    target_archive: str


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_clean_snapshot() -> str:
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("quality-teacher data audit requires a clean worktree")
    if _git("rev-parse", "--abbrev-ref", "HEAD") == "main":
        raise RuntimeError("data audit must execute on a dedicated snapshot branch")
    return _git("rev-parse", "HEAD")


def build_pair_specs(contract: dict[str, Any]) -> tuple[ArchivePair, ...]:
    """Resolve only the frozen trainval members; never enumerate test members."""
    sources = contract["split_policy"]["expected_internal_trainval_sources"]
    dry_archive = contract["dry_markers"]["local_archive"]
    pairs: list[ArchivePair] = []
    for device, archive_root in (
        ("fulltone", "Fulltone-FullDrive2"),
        ("ampeg", "Ampeg-OptoComp"),
    ):
        declaration = contract["devices"][device]
        archive_setting = declaration["archive_setting"]
        for source_id in sources:
            pairs.append(
                ArchivePair(
                    descriptor=PairDescriptor(
                        device=device,
                        source_id=source_id,
                        upstream_split="trainval",
                        input_member=(
                            f"DRY-with-markers/trainval/{source_id}.input.wav"
                        ),
                        target_member=(
                            f"{archive_root}/trainval/{archive_setting}/"
                            f"{archive_setting}.{source_id}.target.wav"
                        ),
                        source_rate_hz=int(declaration["source_rate_hz"]),
                    ),
                    input_archive=dry_archive,
                    target_archive=declaration["local_archive"],
                )
            )
    bigmuff = contract["devices"]["bigmuff"]
    bigmuff_archive = bigmuff["local_archive"]
    setting = bigmuff["archive_setting"]
    for source_id, published_name in (
        ("published-train", "train"),
        ("published-val", "val"),
    ):
        pairs.append(
            ArchivePair(
                descriptor=PairDescriptor(
                    device="bigmuff",
                    source_id=source_id,
                    upstream_split="trainval",
                    input_member=f"DRY/trainval/{published_name}.input.wav",
                    target_member=(
                        f"ElectroHarmonix-BigMuff/trainval/{setting}/"
                        f"{setting}.{published_name}.target.wav"
                    ),
                    source_rate_hz=int(bigmuff["source_rate_hz"]),
                ),
                input_archive=bigmuff_archive,
                target_archive=bigmuff_archive,
            )
        )
    descriptors = [pair.descriptor for pair in pairs]
    validate_source_isolation(descriptors)
    for descriptor in descriptors:
        if "/trainval/" not in descriptor.input_member:
            raise RuntimeError("input member escaped trainval")
        if "/trainval/" not in descriptor.target_member:
            raise RuntimeError("target member escaped trainval")
        if "/test/" in descriptor.input_member or "/test/" in descriptor.target_member:
            raise RuntimeError("test member was resolved before locks")
    return tuple(pairs)


def _published_md5(path: Path) -> str:
    # Zenodo publishes MD5 here as a transport identifier, not a security boundary.
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return f"md5:{digest.hexdigest()}"


def verify_archives(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    declarations = {"dry_markers": contract["dry_markers"]}
    declarations.update(
        {
            device: contract["devices"][device]
            for device in ("fulltone", "bigmuff", "ampeg")
        }
    )
    verified = {}
    for name, declaration in declarations.items():
        path = ROOT / declaration["local_archive"]
        if not path.is_file():
            raise RuntimeError(f"required development archive is absent: {path}")
        size = path.stat().st_size
        if size != int(declaration["size_bytes"]):
            raise RuntimeError(f"published archive size mismatch: {path}")
        checksum = _published_md5(path)
        if checksum != declaration["published_checksum"]:
            raise RuntimeError(f"published archive MD5 mismatch: {path}")
        verified[name] = {
            "local_archive": declaration["local_archive"],
            "published_checksum": checksum,
            "size_bytes": size,
        }
    return verified


def _read_member(archive: Path, member: str) -> tuple[np.ndarray, int]:
    if "/trainval/" not in member or "/test/" in member:
        raise RuntimeError("waveform access is restricted to trainval")
    with ZipFile(archive) as bundle:
        try:
            payload = bundle.read(member)
        except KeyError as error:
            raise RuntimeError(f"missing frozen trainval member: {member}") from error
    signal, rate = sf.read(io.BytesIO(payload), dtype="float32")
    if signal.ndim != 1:
        raise RuntimeError(f"non-mono frozen trainval member: {member}")
    return np.asarray(signal, dtype=np.float32), int(rate)


def _write_audio_once(path: Path, signal: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prepared data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, 48_000, subtype="FLOAT")


def _prepare_pairs(
    pairs: tuple[ArchivePair, ...], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    if OUTPUT_ROOT.exists():
        raise RuntimeError("quality-teacher prepared data already exists")
    records = []
    physical_samples_read = 0
    audit_config = contract["pair_audit"]
    for pair in pairs:
        descriptor = pair.descriptor
        dry, dry_rate = _read_member(ROOT / pair.input_archive, descriptor.input_member)
        wet, wet_rate = _read_member(
            ROOT / pair.target_archive, descriptor.target_member
        )
        if (
            dry_rate != descriptor.source_rate_hz
            or wet_rate != descriptor.source_rate_hz
        ):
            raise RuntimeError("source rate differs from frozen pair descriptor")
        audit = audit_pair_arrays(
            dry,
            wet,
            sample_rate_hz=descriptor.source_rate_hz,
            marker_search_samples=int(audit_config["marker_search_samples"]),
            marker_minimum_amplitude=float(
                audit_config["marker_minimum_absolute_amplitude"]
            ),
            marker_tolerance_samples=int(
                audit_config["marker_alignment_tolerance_samples"]
            ),
            clipping_threshold=float(audit_config["clipping_absolute_threshold"]),
        )
        prepared = prepare_authorized_pair(
            dry,
            wet,
            source_rate_hz=descriptor.source_rate_hz,
        )
        split = assigned_split(
            descriptor.device, descriptor.source_id, descriptor.upstream_split
        )
        output_dir = OUTPUT_ROOT / descriptor.device / split
        input_path = output_dir / f"{descriptor.source_id}_input.wav"
        target_path = output_dir / f"{descriptor.source_id}_target.wav"
        _write_audio_once(input_path, prepared.input)
        _write_audio_once(target_path, prepared.target)
        physical_samples_read += len(dry) + len(wet)
        records.append(
            {
                "audit": asdict(audit),
                "device": descriptor.device,
                "input_member": descriptor.input_member,
                "input_path": str(input_path.relative_to(ROOT)),
                "normalization": "none",
                "output_samples": len(prepared.input),
                "resampling": (
                    "none"
                    if descriptor.source_rate_hz == 48_000
                    else "joint_resample_poly_kaiser_beta_8.6"
                ),
                "source_id": descriptor.source_id,
                "source_rate_hz": descriptor.source_rate_hz,
                "split": split,
                "target_member": descriptor.target_member,
                "target_path": str(target_path.relative_to(ROOT)),
            }
        )
    return records, physical_samples_read


def _write_success_state(physical_samples_read: int) -> None:
    (CAMPAIGN_DIR / "STATE.md").write_text(
        f"""# État

- Lignée : `AMP-QUALITY-TEACHER-v1`.
- Statut : active, `data_audit` passé; `ampeg_slow_value` autorisé.
- Runs scientifiques : 0.
- Gates évalués : 2 (`preflight=passed`, `data_audit=passed`).
- Échantillons physiques de développement lus : {physical_samples_read}.
- Waveforms de test lus : 0.
- Archives Rodent/Fuzzy Logic téléchargées : non.
- Blackstar, UA1176 et `EXTERNAL_REPORT_ONLY` : fermés.
- Prochaine action autorisée : les deux trajectoires Ampeg seed 0, sans ouvrir
  Fulltone/Big Muff scientifique ni aucun `test` avant le gate lent.
""",
        encoding="utf-8",
    )
    (CAMPAIGN_DIR / "HANDOFF.md").write_text(
        """# Handoff

Le préflight et `data_audit` ont passé. Exécuter uniquement les deux
trajectoires Ampeg seed 0 (`s4_tfilm_wavenet_x2_teacher` et
`wavenet_x2_teacher_fast_only`). Ne lancer aucun autre dispositif avant le gate
lent; Rodent/Fuzzy Logic et tous les membres `test` restent fermés.
""",
        encoding="utf-8",
    )


def main() -> int:
    execution_commit = _assert_clean_snapshot()
    decisions = gate_decisions(GATE_LEDGER)
    validate_stage_authorization("data_audit", decisions)
    if EVIDENCE_PATH.exists() or INVALID_PATH.exists():
        raise RuntimeError("quality-teacher data audit is already frozen")
    contract = load_data_contract(ROOT)
    pairs = build_pair_specs(contract)
    archives = verify_archives(contract)
    try:
        records, physical_samples_read = _prepare_pairs(pairs, contract)
    except BaseException as error:
        evidence = {
            "campaign_version": CAMPAIGN_VERSION,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "execution_commit": execution_commit,
            "physical_audio_samples_read_before_failure": None,
            "scientific_runs_launched": 0,
            "stage": "data_audit",
            "status": "invalid",
            "test_waveform_samples_read": 0,
        }
        write_new_json(INVALID_PATH, evidence)
        append_gate_event(
            GATE_LEDGER,
            {
                "campaign_version": CAMPAIGN_VERSION,
                "evidence_path": str(INVALID_PATH.relative_to(ROOT)),
                "stage": "data_audit",
                "status": "invalid",
            },
        )
        raise
    split_groups: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    for record in records:
        split_groups[record["split"]].append(
            {"device": record["device"], "source_id": record["source_id"]}
        )
    manifest = {
        "archives": archives,
        "campaign_version": CAMPAIGN_VERSION,
        "files": records,
        "physical_audio_samples_read": physical_samples_read,
        "test_member_paths_resolved": 0,
        "test_waveform_samples_read": 0,
    }
    split_manifest = {
        "campaign_version": CAMPAIGN_VERSION,
        "cross_split_source_overlap": False,
        "groups": split_groups,
        "test": "sealed_not_materialized",
    }
    write_new_json(MANIFEST_PATH, manifest)
    write_new_json(SPLIT_PATH, split_manifest)
    evidence = {
        "archive_checks": archives,
        "campaign_version": CAMPAIGN_VERSION,
        "development_pair_count": len(records),
        "execution_commit": execution_commit,
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "physical_audio_samples_read": physical_samples_read,
        "scientific_runs_launched": 0,
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
        "stage": "data_audit",
        "status": "passed",
        "test_member_paths_resolved": 0,
        "test_waveform_samples_read": 0,
    }
    write_new_json(EVIDENCE_PATH, evidence)
    append_gate_event(
        GATE_LEDGER,
        {
            "campaign_version": CAMPAIGN_VERSION,
            "evidence_path": str(EVIDENCE_PATH.relative_to(ROOT)),
            "stage": "data_audit",
            "status": "passed",
        },
    )
    maturity_path = CAMPAIGN_DIR / "MATURITY.json"
    maturity = json.loads(maturity_path.read_text(encoding="utf-8"))
    maturity.update(
        {
            "current_stage": "ampeg_slow_value",
            "development_waveform_samples_read": physical_samples_read,
            "gates_evaluated": 2,
            "status": "active_data_audit_passed",
        }
    )
    replace_json(maturity_path, maturity)
    _write_success_state(physical_samples_read)
    print(json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
