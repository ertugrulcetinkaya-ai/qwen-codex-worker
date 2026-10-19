import json
import os
import tempfile
import unittest
from pathlib import Path

from qwen_omlx_auth import api_headers, load_api_key


class OmlxApiKeyTests(unittest.TestCase):
    def write_settings(self, directory: str, document: object) -> Path:
        path = Path(directory) / "settings.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def test_loads_only_api_key_from_private_settings_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_settings(
                directory,
                {"auth": {"api_key": "unit-test-key", "secret_key": "not-the-api-key"}},
            )

            self.assertEqual(load_api_key(path), "unit-test-key")
            self.assertEqual(api_headers(path), {"Authorization": "Bearer unit-test-key"})

    def test_rejects_group_or_world_readable_settings_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_settings(directory, {"auth": {"api_key": "unit-test-key"}})
            os.chmod(path, 0o640)

            with self.assertRaisesRegex(RuntimeError, "OMLX_API_KEY_SETTINGS_PERMISSIONS"):
                load_api_key(path)

    def test_rejects_missing_or_malformed_key_without_echoing_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_settings(directory, {"auth": {"api_key": ""}})

            with self.assertRaisesRegex(RuntimeError, "OMLX_API_KEY_MISSING_OR_INVALID"):
                load_api_key(path)

    def test_rejects_symlinked_settings_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self.write_settings(directory, {"auth": {"api_key": "unit-test-key"}})
            link = Path(directory) / "linked-settings.json"
            link.symlink_to(target)

            with self.assertRaisesRegex(RuntimeError, "OMLX_API_KEY_SETTINGS_UNAVAILABLE"):
                load_api_key(link)


if __name__ == "__main__":
    unittest.main()
