import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs Whisper/pYIN (minutes)")
    config.addinivalue_line("markers", "output: needs a rendered out/ (run `python -m remix.bauen render`)")


@pytest.fixture(scope="session")
def rendered():
    from remix import config
    from remix.mix import master_wav, track_path

    files = [master_wav(), config.OUT / "report.json"] + [track_path(k) for k in config.TRACK_NAMES]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        pytest.skip(f"no render output: {missing[:2]}")
    return True
