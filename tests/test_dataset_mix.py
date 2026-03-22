"""Tests for DuckDB-backed MixedImageDataset percentage-based dataset mixing."""
from __future__ import annotations

import json
import os
import tarfile
import tempfile
import unittest

import duckdb
from PIL import Image


def _make_dummy_images(directory: str, count: int, prefix: str = "img") -> list[str]:
    """Create count tiny JPEG files in directory, return paths."""
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in range(count):
        p = os.path.join(directory, f"{prefix}_{i:04d}.jpg")
        Image.new("RGB", (32, 32), color=(i % 256, 0, 0)).save(p)
        paths.append(p)
    return paths


def _make_tar_archive(archive_path: str, source_dir: str, member_prefix: str = "") -> None:
    """Create a tar.gz archive from files in source_dir."""
    with tarfile.open(archive_path, "w:gz") as tf:
        for name in sorted(os.listdir(source_dir)):
            full = os.path.join(source_dir, name)
            arcname = os.path.join(member_prefix, name) if member_prefix else name
            tf.add(full, arcname=arcname)


def _setup_db(db_path: str, sources: dict[str, list[tuple[str, str, str]]]) -> None:
    """Create a DuckDB with images table and valid_images view.

    sources: {source_name: [(local_path, source_split), ...]}
    Or for convenience: {source_name: [local_path, ...]} (split defaults to "train")
    """
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE images (
            image_id VARCHAR,
            source_name VARCHAR,
            source_split VARCHAR,
            local_path VARCHAR,
            decode_ok BOOLEAN,
            drop_reason VARCHAR
        )
    """)
    con.execute(
        "CREATE VIEW valid_images AS "
        "SELECT * FROM images WHERE decode_ok = true AND drop_reason IS NULL"
    )
    idx = 0
    for source_name, entries in sources.items():
        for entry in entries:
            if isinstance(entry, tuple):
                path, split = entry
            else:
                path, split = entry, "train"
            con.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, true, NULL)",
                [f"{source_name}:{idx}", source_name, split, path],
            )
            idx += 1
    con.close()


class TestMixedImageDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

        # Create filesystem image sources
        self.src_a_dir = os.path.join(self.tmpdir, "images", "source_a")
        self.src_b_dir = os.path.join(self.tmpdir, "images", "source_b")
        a_paths = _make_dummy_images(self.src_a_dir, 100)
        b_paths_train = _make_dummy_images(
            os.path.join(self.tmpdir, "images", "source_b_train"), 150, prefix="trn"
        )
        b_paths_val = _make_dummy_images(
            os.path.join(self.tmpdir, "images", "source_b_val"), 50, prefix="val"
        )

        # Create a tar.gz archive source
        tar_src_dir = os.path.join(self.tmpdir, "tar_src")
        _make_dummy_images(tar_src_dir, 50, prefix="arc")
        self.archive_path = os.path.join(self.tmpdir, "archives", "source_c.tar.gz")
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
        _make_tar_archive(self.archive_path, tar_src_dir, member_prefix="train")
        c_paths = [f"{self.archive_path}::train/arc_{i:04d}.jpg" for i in range(50)]

        # Set up DuckDB — source_b has train+val splits
        self.db_path = os.path.join(self.tmpdir, "test.duckdb")
        _setup_db(self.db_path, {
            "source_a": a_paths,  # all default to split="train"
            "source_b": (
                [(p, "train") for p in b_paths_train]
                + [(p, "val") for p in b_paths_val]
            ),
            "source_c": c_paths,  # defaults to split="train"
        })
        self.staging_dir = os.path.join(self.tmpdir, "staging")

    def _make_dataset(self, mix, **kwargs):
        from torchvision import transforms
        from train.dino_ssl import MixedImageDataset

        tfm = transforms.Compose([transforms.ToTensor()])
        kwargs.setdefault("staging_dir", self.staging_dir)
        return MixedImageDataset(self.db_path, mix, tfm, **kwargs)

    def test_full_100_percent(self):
        ds = self._make_dataset({"source_a": 100, "source_b": 100})
        self.assertEqual(len(ds), 300)
        self.assertEqual(ds.source_counts["source_a"][0], 100)  # total
        self.assertEqual(ds.source_counts["source_a"][1], 100)  # kept
        self.assertEqual(ds.source_counts["source_b"][0], 200)
        self.assertEqual(ds.source_counts["source_b"][1], 200)

    def test_partial_percentages(self):
        ds = self._make_dataset({"source_a": 50, "source_b": 25})
        self.assertEqual(ds.source_counts["source_a"][1], 50)
        self.assertEqual(ds.source_counts["source_b"][1], 50)
        self.assertEqual(len(ds), 100)

    def test_small_percentage_floors_to_one(self):
        ds = self._make_dataset({"source_a": 1})
        # 1% of 100 = 1
        self.assertEqual(ds.source_counts["source_a"][1], 1)
        self.assertEqual(len(ds), 1)

    def test_zero_percentage_skipped(self):
        ds = self._make_dataset({"source_a": 0, "source_b": 100})
        self.assertNotIn("source_a", ds.source_counts)
        self.assertEqual(len(ds), 200)

    def test_over_100_raises(self):
        with self.assertRaises(ValueError):
            self._make_dataset({"source_a": 150})

    def test_missing_source_raises(self):
        with self.assertRaises(ValueError):
            self._make_dataset({"nonexistent": 50})

    def test_deterministic_with_same_seed(self):
        ds1 = self._make_dataset({"source_a": 30, "source_b": 40}, seed=42)
        ds2 = self._make_dataset({"source_a": 30, "source_b": 40}, seed=42)
        self.assertEqual(sorted(ds1.items), sorted(ds2.items))

    def test_different_seed_gives_different_subset(self):
        ds1 = self._make_dataset({"source_b": 25}, seed=1)
        ds2 = self._make_dataset({"source_b": 25}, seed=2)
        self.assertEqual(len(ds1), len(ds2))
        self.assertNotEqual(sorted(ds1.items), sorted(ds2.items))

    def test_adding_source_doesnt_change_other(self):
        """Per-source seeded RNG: adding source_b shouldn't change source_a's subset."""
        ds1 = self._make_dataset({"source_a": 50}, seed=7)
        ds2 = self._make_dataset({"source_a": 50, "source_b": 30}, seed=7)
        a_items_1 = sorted(ds1.items)
        a_items_2 = sorted(p for p in ds2.items if "source_a" in p)
        self.assertEqual(a_items_1, a_items_2)

    def test_max_images_cap(self):
        ds = self._make_dataset({"source_a": 100, "source_b": 100}, max_images=50)
        self.assertEqual(len(ds), 50)

    def test_getitem_returns_tensor(self):
        import torch
        ds = self._make_dataset({"source_a": 10})
        out = ds[0]
        self.assertIsInstance(out, torch.Tensor)

    def test_json_parsing_roundtrip(self):
        """Simulate the argparse JSON string path."""
        mix_str = '{"source_a": 20, "source_b": 80}'
        mix = json.loads(mix_str)
        ds = self._make_dataset(mix)
        self.assertEqual(ds.source_counts["source_a"][1], 20)
        self.assertEqual(ds.source_counts["source_b"][1], 160)
        self.assertEqual(len(ds), 180)

    def test_tar_archive_extraction(self):
        """Archive paths with :: should be extracted to staging and loadable."""
        ds = self._make_dataset({"source_c": 100})
        self.assertEqual(ds.source_counts["source_c"][0], 50)   # total
        self.assertEqual(ds.source_counts["source_c"][1], 50)   # kept
        self.assertEqual(ds.source_counts["source_c"][2], 0)    # skipped
        self.assertEqual(len(ds), 50)
        # Verify files were extracted to staging
        for p in ds.items:
            self.assertTrue(os.path.isfile(p), f"Extracted file missing: {p}")

    def test_tar_partial_percentage(self):
        """Subsample works on archive sources too."""
        ds = self._make_dataset({"source_c": 20})
        self.assertEqual(ds.source_counts["source_c"][0], 50)
        self.assertEqual(ds.source_counts["source_c"][1], 10)   # 20% of 50
        self.assertEqual(len(ds), 10)

    def test_tar_getitem_loads_image(self):
        """Extracted archive images are loadable."""
        import torch
        ds = self._make_dataset({"source_c": 10})
        out = ds[0]
        self.assertIsInstance(out, torch.Tensor)

    def test_mixed_fs_and_archive(self):
        """Mix filesystem and archive sources together."""
        ds = self._make_dataset({"source_a": 50, "source_c": 100})
        self.assertEqual(ds.source_counts["source_a"][1], 50)
        self.assertEqual(ds.source_counts["source_c"][1], 50)
        self.assertEqual(len(ds), 100)

    def test_tar_extraction_caching(self):
        """Second dataset creation should hit the staging cache, not re-extract."""
        ds1 = self._make_dataset({"source_c": 100})
        # Verify staging dir has files
        staging_files_before = set()
        for root, _, files in os.walk(self.staging_dir):
            for f in files:
                staging_files_before.add(os.path.join(root, f))
        self.assertTrue(len(staging_files_before) > 0)

        # Create again - should reuse cache
        ds2 = self._make_dataset({"source_c": 100})
        self.assertEqual(sorted(ds1.items), sorted(ds2.items))

    # --- split filtering tests ---

    def test_split_filter_train_only(self):
        """source_b:train should only get the 150 train images."""
        ds = self._make_dataset({"source_b:train": 100})
        self.assertEqual(ds.source_counts["source_b:train"][0], 150)
        self.assertEqual(ds.source_counts["source_b:train"][1], 150)
        self.assertEqual(len(ds), 150)

    def test_split_filter_val_only(self):
        """source_b:val should only get the 50 val images."""
        ds = self._make_dataset({"source_b:val": 100})
        self.assertEqual(ds.source_counts["source_b:val"][0], 50)
        self.assertEqual(ds.source_counts["source_b:val"][1], 50)
        self.assertEqual(len(ds), 50)

    def test_split_filter_with_percentage(self):
        ds = self._make_dataset({"source_b:train": 20})
        self.assertEqual(ds.source_counts["source_b:train"][0], 150)
        self.assertEqual(ds.source_counts["source_b:train"][1], 30)  # 20% of 150
        self.assertEqual(len(ds), 30)

    def test_unqualified_gets_all_splits(self):
        """Without :split, all splits are included."""
        ds = self._make_dataset({"source_b": 100})
        self.assertEqual(ds.source_counts["source_b"][0], 200)  # 150 train + 50 val
        self.assertEqual(len(ds), 200)

    def test_mix_qualified_and_unqualified(self):
        """Can mix source_a (no split qualifier) with source_b:train."""
        ds = self._make_dataset({"source_a": 100, "source_b:train": 100})
        self.assertEqual(len(ds), 250)  # 100 + 150

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            self._make_dataset({"source_b:nonexistent": 50})

    def test_error_message_shows_available_splits(self):
        try:
            self._make_dataset({"source_b:nonexistent": 50})
        except ValueError as e:
            msg = str(e)
            self.assertIn("source_b:train", msg)
            self.assertIn("source_b:val", msg)


if __name__ == "__main__":
    unittest.main()
