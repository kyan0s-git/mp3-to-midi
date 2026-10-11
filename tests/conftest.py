import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="session")
def song():
    """A short synthetic song: (mix, ground truth, stems)."""
    from synthetic import make_song

    return make_song(seed=0, key_name="C major", bpm=100, bars=4)
