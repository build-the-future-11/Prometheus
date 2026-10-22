import importlib.util
import hashlib
import io
import json
import statistics
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_s3_gpu_study.py"
ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("run_s3_gpu_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# Match normal import semantics before executing dataclass declarations.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VERIFIER_SCRIPT = ROOT / "scripts" / "verify_s3_hosted_archive.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location("verify_s3_hosted_archive", VERIFIER_SCRIPT)
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
assert VERIFIER_SPEC.loader is not None
sys.modules[VERIFIER_SPEC.name] = VERIFIER
VERIFIER_SPEC.loader.exec_module(VERIFIER)


class RegisteredResultClassificationTests(unittest.TestCase):
    def classify(self, paired_count, mean_improvement, wins):
        return MODULE.classify_registered_result(
            paired_count=paired_count,
            required_count=5,
            mean_improvement=mean_improvement,
            wins=wins,
            minimum_improvement=0.02,
            minimum_wins=4,
        )

    def test_supported_requires_both_registered_clauses(self):
        self.assertEqual(self.classify(5, 0.02, 4), "SUPPORTED")

    def test_mixed_retains_one_clause_outcomes(self):
        self.assertEqual(self.classify(5, 0.03, 3), "MIXED")
        self.assertEqual(self.classify(5, 0.01, 4), "MIXED")

    def test_negative_requires_complete_evidence_and_both_clauses_fail(self):
        self.assertEqual(self.classify(5, 0.01, 3), "NEGATIVE")

    def test_incomplete_pairs_are_inconclusive(self):
        self.assertEqual(self.classify(4, 0.10, 4), "INCONCLUSIVE")
        self.assertEqual(self.classify(0, None, 0), "INCONCLUSIVE")


class KaggleReleaseContractTests(unittest.TestCase):
    def test_kernel_metadata_uses_typed_private_gpu_settings(self):
        metadata_path = ROOT / "kaggle" / "kernel-metadata.json"
        if not metadata_path.exists():
            self.skipTest("kernel metadata is intentionally outside the hosted source bundle")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertIs(metadata["is_private"], True)
        self.assertIs(metadata["enable_gpu"], True)
        self.assertIs(metadata["enable_internet"], False)
        self.assertEqual(
            metadata["dataset_sources"],
            [
                "ryangomez010/derived-learning-s3-frozen-source",
                "camnugent/california-housing-prices",
            ],
        )

    def test_release_bundle_is_hash_bound_and_contains_only_source(self):
        release = ROOT / "releases" / "s3-r0"
        bundle = release / "dlml-s3-source.bundle"
        sidecar = release / "dlml-s3-source.sha256"
        if not bundle.exists() or not sidecar.exists():
            self.skipTest("release artifacts are intentionally outside the hosted source bundle")

        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
        self.assertEqual(actual, expected)

        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            self.assertNotEqual(names, [])
            for member in members:
                path = Path(member.name)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertFalse(any(part == ".DS_Store" or part.startswith("._") for part in path.parts))
                self.assertFalse(member.issym() or member.islnk() or member.isdev())

                if member.isfile():
                    local = ROOT / member.name
                    self.assertTrue(local.is_file(), member.name)
                    archived_hash = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                    self.assertEqual(archived_hash, hashlib.sha256(local.read_bytes()).hexdigest(), member.name)


class HostedArchiveVerificationTests(unittest.TestCase):
    def test_verifier_recomputes_complete_raw_result_matrix(self):
        config_path = ROOT / "configs" / "s3_real_data_gpu.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_root = "runs/kaggle-dlml-s3-r0-r1"
        selected = {
            str(seed): {method: candidates[0] for method, candidates in config["candidate_grid"].items()}
            for seed in config["seeds"]
        }

        attempts = []
        for seed in config["seeds"]:
            for method, candidates in config["candidate_grid"].items():
                for index, parameters in enumerate(candidates):
                    attempts.append(
                        {
                            "seed": seed,
                            "method": method,
                            "parameters": parameters,
                            "status": "SUCCEEDED",
                            "test_metrics_present": False,
                            "development": {"standardized_rmse": 1.0 + index},
                        }
                    )

        method_rmse = {
            "joint_block_mlp": 100.0,
            "residual_block_mlp": 95.0,
            "ridge": 90.0,
            "hist_gradient_boosting": 80.0,
        }
        results = []
        for seed in config["seeds"]:
            for method in config["candidate_grid"]:
                rmse = method_rmse[method] + (seed - config["seeds"][0]) * 0.01
                results.append(
                    {
                        "seed": seed,
                        "method": method,
                        "parameters": selected[str(seed)][method],
                        "status": "SUCCEEDED",
                        "split": "protected_test",
                        "test": {
                            "standardized_rmse": rmse / 100.0,
                            "rmse": rmse,
                            "wall_seconds": 1.0,
                            "peak_gpu_bytes": 1024,
                        },
                    }
                )

        pairs = []
        for seed in config["seeds"]:
            joint = next(row["test"]["rmse"] for row in results if row["seed"] == seed and row["method"] == "joint_block_mlp")
            residual = next(row["test"]["rmse"] for row in results if row["seed"] == seed and row["method"] == "residual_block_mlp")
            pairs.append(
                {
                    "seed": seed,
                    "joint_rmse": joint,
                    "residual_rmse": residual,
                    "relative_improvement": (joint - residual) / joint,
                }
            )
        method_means = {
            method: statistics.fmean(row["test"]["rmse"] for row in results if row["method"] == method)
            for method in config["candidate_grid"]
        }
        summary = {
            "study_id": config["study_id"],
            "paired": pairs,
            "paired_count": len(pairs),
            "paired_wins": len(pairs),
            "mean_relative_improvement": statistics.fmean(row["relative_improvement"] for row in pairs),
            "method_mean_rmse": method_means,
            "registered_result": "SUPPORTED",
            "broad_superiority_claim_allowed": False,
            "independent_reproduction": False,
        }

        payloads = {
            "selection_attempts.jsonl": b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in attempts
            ),
            "results.jsonl": b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in results
            ),
            "selection.json": (json.dumps({"uses_test_targets": False, "selected": selected}, sort_keys=True) + "\n").encode(),
            "summary.json": (json.dumps(summary, sort_keys=True) + "\n").encode(),
            "environment.json": b"{}\n",
            "resource_admission.json": b'{"admitted": true}\n',
        }
        for seed in config["seeds"]:
            payloads[f"splits/seed-{seed}.json"] = (
                json.dumps({"seed": seed, "test_targets_used_for_selection": False}, sort_keys=True) + "\n"
            ).encode()

        manifest = {
            "study_id": config["study_id"],
            "status": "COMPLETE",
            "dataset_path": "/kaggle/input/california-housing-prices/housing.csv",
            "dataset_sha256": "0" * 64,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256((ROOT / config["protocol_path"]).read_bytes()).hexdigest(),
            "outputs": {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(payloads.items())},
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            archive_path = directory_path / "kaggle-dlml-s3-r0-r1.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for relative, payload in payloads.items():
                    info = tarfile.TarInfo(f"{run_root}/{relative}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                manifest_payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
                info = tarfile.TarInfo(f"{run_root}/manifest.json")
                info.size = len(manifest_payload)
                archive.addfile(info, io.BytesIO(manifest_payload))
            sidecar_path = directory_path / f"{archive_path.name}.sha256"
            sidecar_path.write_text(
                f"{hashlib.sha256(archive_path.read_bytes()).hexdigest()}  {archive_path.name}\n",
                encoding="utf-8",
            )
            receipt = VERIFIER.verify_archive(archive_path, sidecar_path, config_path, ROOT)

        self.assertEqual(receipt["verification_status"], "VERIFIED")
        self.assertEqual(receipt["registered_result"], "SUPPORTED")
        self.assertEqual(receipt["selection_attempt_count"], 50)
        self.assertEqual(receipt["protected_result_count"], 20)
        self.assertEqual(receipt["paired_count"], 5)


if __name__ == "__main__":
    unittest.main()
