"""bouquet.animate: archive discovery across studies and the shared frame limits."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from bouquet.animate import _share_limits, shot_archives  # noqa: E402


def test_shot_archives_numeric_order_across_studies(tmp_path):
    for study, names in (("a", ["7_996_base.h5", "7_2456_base.h5"]), ("b", ["7_1590_base.h5", "7_996_base.h5"])):
        d = tmp_path / study / ("c000" if study == "b" else "") / "7" / "bouquet"
        d.mkdir(parents=True)
        for n in names + ["index.json"]:
            (d / n).write_text("")
    with pytest.warns(UserWarning, match=r"\(996, 'base'\)"):
        a = shot_archives([str(tmp_path / "a"), str(tmp_path / "b")], 7)
    assert [(t, s) for t, _, _, s in a] == [(996, "a"), (996, "b"), (1590, "b"), (2456, "a")]


def test_share_limits():
    frames = []
    for y in ((0, 1), (-2, 0.5)):
        fig, ax = plt.subplots()
        ax.plot([0, 1], y)
        ax.invert_yaxis()
        frames.append([fig])
    _share_limits(frames)
    lims = [f[0].axes[0].get_ylim() for f in frames]
    assert lims[0] == lims[1] and lims[0][0] > lims[0][1] and lims[0][1] <= -2 and lims[0][0] >= 1
    plt.close("all")
