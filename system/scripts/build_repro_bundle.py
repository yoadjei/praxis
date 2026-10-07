# -*- coding: utf-8 -*-
"""Build and verify reproducibility bundles for PRAXIS runs.

A bundle packages one run's frozen config, manifest, all artefacts, and a
checksum index, so a reader can verify computation offline against the bytes
that produced the numbers. Media files are excluded because they are personal
data under Act 843 and the bundle is for verifying computation, not
redistributing the corpus.

Usage:
    python scripts/build_repro_bundle.py --out bundle.zip
    python scripts/build_repro_bundle.py --out bundle.zip --run <run_id>
    python scripts/build_repro_bundle.py --verify bundle.zip

Exit codes: 0 success, 1 failed, 2 usage, 3 hash mismatch on verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.config import load_config
from praxis.phases.artefacts import sha256_of
from praxis.phases.chain import completed_runs

CHUNK = 1024 * 1024


def get_run_directory(run_id: str | None = None) -> Path:
    """Find a run directory by ID or return the latest run.

    Raises RuntimeError if no run is found.
    """
    config = load_config()
    root = config.paths.run_outputs

    if run_id is not None:
        path = root / run_id
        if not path.is_dir():
            raise RuntimeError(f"run {run_id!r} not found in {root}")
        return path

    # Find the latest run that produced a bundle_index.json.
    runs = completed_runs(config, "11")
    if not runs:
        raise RuntimeError(
            f"no completed run of phase 11 found in {root}; "
            "run: python scripts/run_phase.py --phase 11")

    return runs[0].directory


def read_bundle_index(run_dir: Path) -> dict[str, object]:
    """Load the bundle index JSON written by phase 11.

    Raises RuntimeError if the file does not exist or is unreadable.
    """
    index_path = run_dir / "bundle_index.json"
    if not index_path.is_file():
        raise RuntimeError(
            f"phase 11 output bundle_index.json not found in {run_dir}; "
            "run: python scripts/run_phase.py --phase 11")

    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read bundle_index.json: {exc}") from exc


def read_manifest(run_dir: Path) -> dict[str, object]:
    """Load the run manifest written during phase execution.

    Raises RuntimeError if the file does not exist or is unreadable.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest.json not found in {run_dir}")

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read manifest.json: {exc}") from exc


def get_git_commit() -> str | None:
    """Get the current commit hash, or None if not in a repo or git fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_torch_version() -> str:
    """Get torch version, or a string indicating it is not installed."""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not installed"


def build_bundle(run_dir: Path, output_path: Path) -> int:
    """Build a reproducibility bundle from a run directory.

    Verifies all file hashes match the index before creating the zip.
    Returns 0 on success, 1 on error.
    """
    try:
        bundle_index = read_bundle_index(run_dir)
        manifest = read_manifest(run_dir)

        # Verify every file in the index has a matching hash on disk.
        print(f"verifying {len(bundle_index.get('files', []))} files...")
        for file_spec in bundle_index.get("files", []):
            file_path_rel = file_spec.get("path", "")
            expected_sha = file_spec.get("sha256", "")
            file_path = run_dir / file_path_rel

            if not file_path.is_file():
                print(f"error: {file_path_rel} not found", file=sys.stderr)
                return 1

            actual_sha = sha256_of(file_path)
            if actual_sha != expected_sha:
                print(
                    f"error: {file_path_rel} hashes to {actual_sha[:12]}, "
                    f"index expects {expected_sha[:12]}",
                    file=sys.stderr)
                return 1

        # Collect metadata.
        run_id = manifest.get("run_id", "unknown")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}."
        python_version += f"{sys.version_info.micro}"
        torch_version = get_torch_version()
        git_commit = get_git_commit()
        created_at = datetime.utcnow().isoformat() + "Z"

        # Create bundle with temp file to avoid partial writes.
        with tempfile.NamedTemporaryFile(
                suffix=".zip", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with zipfile.ZipFile(temp_path, "w",
                                 zipfile.ZIP_DEFLATED) as bundle:
                # Add core files: manifest, result, and config.
                manifest_path = run_dir / "manifest.json"
                if manifest_path.is_file():
                    bundle.write(manifest_path, arcname="manifest.json")

                result_path = run_dir / "result.json"
                if result_path.is_file():
                    bundle.write(result_path, arcname="result.json")

                # Add all files from the index (config and artefacts).
                for file_spec in bundle_index.get("files", []):
                    file_path_rel = file_spec.get("path", "")
                    file_path = run_dir / file_path_rel
                    bundle.write(file_path, arcname=file_path_rel)

                # Add bundle_index.json itself for verification.
                bundle_index_path = run_dir / "bundle_index.json"
                bundle.write(bundle_index_path, arcname="bundle_index.json")

                # Build the bundle metadata JSON.
                bundle_meta: dict[str, object] = {
                    "created_at": created_at,
                    "run_id": run_id,
                    "python_version": python_version,
                    "torch_version": torch_version,
                    "git_commit": git_commit,
                    "files": bundle_index.get("files", []),
                    "total_files": bundle_index.get("file_count", 0),
                    "total_bytes": bundle_index.get("total_bytes", 0),
                }

                # Write BUNDLE.json.
                bundle_json_str = json.dumps(
                    bundle_meta, indent=2, sort_keys=True) + "\n"
                bundle.writestr("BUNDLE.json", bundle_json_str)

                # Write README.md with verification instructions.
                readme_text = (
                    "# Reproducibility Bundle\n\n"
                    "This archive contains the complete artefacts needed to verify a "
                    "PRAXIS run,\n"
                    "including the frozen config, all intermediate outputs, and "
                    "checksums.\n\n"
                    "## Contents\n\n"
                    "- `config.yaml` - the frozen configuration\n"
                    "- `manifest.json` - run metadata and device information\n"
                    "- `result.json` - phase summary\n"
                    "- `bundle_index.json` - checksums of all artefacts\n"
                    "- `BUNDLE.json` - bundle metadata including Python and PyTorch "
                    "versions\n"
                    "- All split manifests and intermediate outputs named in the index\n\n"
                    "**Media files are excluded** because they are personal data under "
                    "Act 843.\n"
                    "This bundle is for verifying computation, not for redistributing "
                    "the corpus.\n\n"
                    "## Verify the Bundle\n\n"
                    "```bash\n"
                    "python scripts/build_repro_bundle.py --verify bundle.zip\n"
                    "```\n\n"
                    "This will check every file's SHA-256 hash against the index.\n"
                    "Exit code 0 means all hashes match; non-zero means at least one "
                    "does not.\n\n"
                    "## Manual Verification\n\n"
                    "Extract the bundle and compute file hashes using your preferred "
                    "tool:\n\n"
                    "```bash\n"
                    "unzip bundle.zip\n"
                    "sha256sum config.yaml  # compare against BUNDLE.json\n"
                    "```\n"
                )
                bundle.writestr("README.md", readme_text)

            # Move temp file to output location.
            temp_path.replace(output_path)
            print(f"created {output_path}")
            return 0

        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise exc

    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: failed to build bundle: {exc}", file=sys.stderr)
        return 1


def verify_bundle(bundle_path: Path) -> int:
    """Verify all hashes in a bundle match the files inside it.

    Returns 0 if all hashes match, 3 if any mismatch is found, 1 on other
    errors.
    """
    if not bundle_path.is_file():
        print(f"error: {bundle_path} not found", file=sys.stderr)
        return 1

    try:
        with zipfile.ZipFile(bundle_path, "r") as bundle:
            # Read the bundle index to get expected hashes.
            try:
                index_data = bundle.read("bundle_index.json")
                bundle_index = json.loads(index_data.decode("utf-8"))
            except KeyError:
                print(
                    "error: bundle_index.json not found in bundle",
                    file=sys.stderr)
                return 1

            # Verify each file's hash.
            files = bundle_index.get("files", [])
            mismatches = []

            for file_spec in files:
                file_path_rel = file_spec.get("path", "")
                expected_sha = file_spec.get("sha256", "")

                try:
                    file_data = bundle.read(file_path_rel)
                except KeyError:
                    print(
                        f"error: {file_path_rel} not found in bundle",
                        file=sys.stderr)
                    return 1

                # Compute hash of the file data.
                digest = hashlib.sha256()
                digest.update(file_data)
                actual_sha = digest.hexdigest()

                if actual_sha != expected_sha:
                    mismatches.append((file_path_rel, actual_sha, expected_sha))

            if mismatches:
                print(f"error: {len(mismatches)} file(s) have mismatched hashes:",
                      file=sys.stderr)
                for path, actual, expected in mismatches:
                    print(
                        f"  {path}: got {actual[:12]}, expected "
                        f"{expected[:12]}",
                        file=sys.stderr)
                return 3

            print(f"{bundle_path}: verified {len(files)} files, all hashes match")
            return 0

    except (zipfile.BadZipFile, OSError) as exc:
        print(f"error: failed to read bundle: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: verification failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """Parse arguments and dispatch to build or verify."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--out", type=Path,
        help="output bundle path (requires building a new bundle)")
    group.add_argument(
        "--verify", type=Path,
        help="verify an existing bundle")

    parser.add_argument(
        "--run", type=str,
        help="run ID to bundle (uses latest if not specified)")

    args = parser.parse_args()

    if args.out is not None:
        # Build mode.
        try:
            run_dir = get_run_directory(args.run)
            print(f"bundling run from {run_dir}")
            return build_bundle(run_dir, args.out)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 1

    elif args.verify is not None:
        # Verify mode.
        return verify_bundle(args.verify)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
