# Copyright (c) 2026 ain-soph

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLazyAlignment(unittest.TestCase):
    def run_python(self, code):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_model_api_import_does_not_load_alignment_dependencies(self):
        result = self.run_python(
            "import sys,json; import mlx_whisper; "
            "from mlx_whisper.transcribe import ModelHolder; "
            "print(json.dumps({"
            "'transcribe':callable(mlx_whisper.transcribe),"
            "'model_loader':callable(ModelHolder.get_model),"
            "'timing':'mlx_whisper.timing' in sys.modules,"
            "'numba':'numba' in sys.modules,"
            "'scipy':'scipy' in sys.modules}))"
        )
        self.assertTrue(result["transcribe"])
        self.assertTrue(result["model_loader"])
        self.assertFalse(result["timing"])
        self.assertFalse(result["numba"])
        self.assertFalse(result["scipy"])

    def test_explicit_alignment_import_still_works(self):
        result = self.run_python(
            "import sys,json; import mlx_whisper; "
            "from mlx_whisper.timing import add_word_timestamps; "
            "print(json.dumps({"
            "'alignment':callable(add_word_timestamps),"
            "'numba':'numba' in sys.modules,"
            "'scipy':'scipy' in sys.modules}))"
        )
        self.assertTrue(result["alignment"])
        self.assertTrue(result["numba"])
        self.assertTrue(result["scipy"])


if __name__ == "__main__":
    unittest.main()
