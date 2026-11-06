import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from artifacts import install_archive, sha256, verify_adapter


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "source.bin"
        source.write_bytes(b"test adapter bytes")
        self.descriptor = {"id": "test-model", "files": {"adapter_model.safetensors": {
            "bytes": source.stat().st_size, "sha256": sha256(source)}}}
        self.archive = self.root / "model.tar.gz"
        with tarfile.open(self.archive, "w:gz") as handle:
            handle.add(source, arcname="adapter_model.safetensors")
        self.descriptor["archive_sha256"] = sha256(self.archive)

    def tearDown(self):
        self.temporary.cleanup()

    def test_verified_install_and_corruption_detection(self):
        result = install_archive(self.archive, self.root / "adapters", self.descriptor)
        verify_adapter(result, self.descriptor)
        self.assertEqual(install_archive(self.archive, self.root / "adapters", self.descriptor), result)
        (result / "adapter_model.safetensors").write_bytes(b"corruption")
        with self.assertRaises(ValueError):
            verify_adapter(result, self.descriptor)

    def test_rejects_checksum_mismatch_before_extraction(self):
        self.descriptor["archive_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "archive checksum"):
            install_archive(self.archive, self.root / "adapters", self.descriptor)
        self.assertFalse((self.root / "adapters").exists())

    def test_rejects_path_traversal_and_links(self):
        for name, kind in [("../outside", tarfile.REGTYPE), ("adapter_model.safetensors", tarfile.SYMTYPE)]:
            with tarfile.open(self.archive, "w:gz") as handle:
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind == tarfile.SYMTYPE:
                    member.linkname = "../outside"
                    handle.addfile(member)
                else:
                    member.size = 1
                    handle.addfile(member, io.BytesIO(b"x"))
            self.descriptor["archive_sha256"] = sha256(self.archive)
            with self.assertRaises(ValueError):
                install_archive(self.archive, self.root / "adapters", self.descriptor)
        self.assertFalse((self.root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
