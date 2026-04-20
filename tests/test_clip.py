from pathlib import Path

from owedits.clip import clip_name
from owedits.events import HighlightEvent


def test_clip_name_format():
    ev = HighlightEvent(
        video=Path("/vids/match_03.mp4"),
        kind="multikill",
        kill_count=3,
        first_kill_t=442.5,
        last_kill_t=444.2,
        start_s=432.5,
        end_s=449.2,
    )
    assert clip_name(ev) == "match_03__multikill_3x_07-22.mp4"


def test_clip_name_team_wipe():
    ev = HighlightEvent(
        video=Path("foo.mp4"),
        kind="team_wipe",
        kill_count=5,
        first_kill_t=61.9,
        last_kill_t=66.0,
        start_s=51.9,
        end_s=71.0,
    )
    assert clip_name(ev) == "foo__team_wipe_5x_01-01.mp4"
