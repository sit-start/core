import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sitstart.util.run import Output, run


def test_run_with_std_output(capfd):
    cmd = ["echo", "Hello, World!"]
    result = run(cmd, output=Output.STD)

    captured = capfd.readouterr()
    assert captured.out == "Hello, World!\n"
    assert captured.err == ""
    assert result.returncode == 0


def test_run_with_file_output(caplog):
    caplog.set_level(logging.INFO)

    cmd = ["echo", "Hello, World!"]
    result = run(cmd, output=Output.FILE)

    out = re.search(f"Writing {cmd[0]!r} command output to (.*)$", caplog.text)
    assert out and len(out.groups()) == 1
    out_path = out.groups()[0].strip()

    captured = Path(out_path).read_text().strip()
    assert captured == "Hello, World!"
    assert result.returncode == 0


def test_run_with_capture_output():
    cmd = ["echo", "Hello, World!"]
    result = run(cmd, output=Output.CAPTURE)

    assert result.stdout.decode(sys.stdout.encoding).strip() == "Hello, World!"
    assert result.stderr == b""
    assert result.returncode == 0


def test_run_with_capture_output_error():
    cmd = ["invalid_command"]
    with pytest.raises(FileNotFoundError, match=f".*{cmd[0]}.*"):
        run(cmd, output=Output.CAPTURE)

    cmd = ["ls", "nonexistent_file"]
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(subprocess.CalledProcessError):
            run(cmd, output=Output.CAPTURE, cwd=temp_dir)
