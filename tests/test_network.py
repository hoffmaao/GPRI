"""Epoch/pair bookkeeping and the SBAS design matrices."""
from datetime import datetime

import numpy as np
import pytest

from gpri.network import (Network, parse_epoch, read_itab, read_slc_tab,
                          read_tab, scene_id, write_itab)


def test_parse_epoch_from_a_gpri_path():
    assert parse_epoch("slc/20170803_222136u.slc") == datetime(2017, 8, 3, 22, 21, 36)


def test_parse_epoch_rejects_junk():
    with pytest.raises(ValueError, match="timestamp"):
        parse_epoch("not_a_scene.slc")


def test_scene_id_strips_directory_and_suffix():
    assert scene_id("diff0/20170803_222136u.diff") == "20170803_222136u"


def test_read_tab_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "SLC_tab"
    f.write_text("# header\n\nslc/a.slc slc/a.slc.par\nslc/b.slc slc/b.slc.par\n")
    assert read_tab(f) == [["slc/a.slc", "slc/a.slc.par"],
                           ["slc/b.slc", "slc/b.slc.par"]]


def test_read_slc_tab_infers_missing_par_column(tmp_path):
    f = tmp_path / "SLC_tab"
    f.write_text("slc/a.slc\n")
    imgs, pars = read_slc_tab(f)
    assert imgs == ["slc/a.slc"] and pars == ["slc/a.slc.par"]


def _itab(tmp_path, text):
    f = tmp_path / "itab"
    f.write_text(text)
    return f


def test_itab_is_converted_to_zero_based(tmp_path):
    p = read_itab(_itab(tmp_path, "   1    2    1    1\n   2    3    2    1\n"))
    assert np.array_equal(p, [[0, 1], [1, 2]])


def test_itab_drops_the_self_pair(tmp_path):
    """GAMMA's stacking scripts emit a leading `1 1`; it carries no phase."""
    p = read_itab(_itab(tmp_path, "   1    1    1    1\n   1    2    2    1\n"))
    assert np.array_equal(p, [[0, 1]])


def test_itab_honours_the_disable_flag(tmp_path):
    p = read_itab(_itab(tmp_path, "   1    2    1    0\n   2    3    2    1\n"))
    assert np.array_equal(p, [[1, 2]])


def test_itab_roundtrip(tmp_path):
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    f = tmp_path / "itab_out"
    write_itab(f, pairs)
    assert np.array_equal(read_itab(f), pairs)


def _net(n=4, pairs=None):
    epochs = [datetime(2017, 8, 3, 22, 0, 0) for _ in range(n)]
    epochs = [datetime(2017, 8, 3, 22, 0, 0).replace(minute=2 * i) for i in range(n)]
    if pairs is None:
        pairs = [(i, i + 1) for i in range(n - 1)]
    return Network(epochs, pairs)


def test_network_rejects_out_of_range_pair():
    with pytest.raises(ValueError, match="beyond"):
        Network([datetime(2017, 1, 1)] * 2, [(0, 5)])


def test_times_are_days_from_the_first_epoch():
    net = _net(4)
    assert net.times[0] == 0.0
    assert net.times[1] == pytest.approx(2 / (60 * 24))


def test_temporal_baselines_match_pairs():
    net = _net(4)
    assert np.allclose(net.temporal_baselines(), 2 / (60 * 24))


def test_daisy_chain_is_connected():
    assert _net(5).is_connected()


def test_disconnected_network_is_reported():
    net = Network([datetime(2017, 1, 1)] * 4, [(0, 1), (2, 3)])
    assert not net.is_connected()
    assert [sorted(c) for c in net.components()] == [[0, 1], [2, 3]]


def test_design_matrix_row_is_dj_minus_di():
    """Row for pair (i, j) must encode d_j - d_i, with the reference dropped."""
    net = _net(3)                      # pairs (0,1), (1,2); reference 0
    G = net.design_matrix(reference=0)
    assert G.shape == (2, 2)           # 2 pairs, 2 free epochs (1 and 2)
    assert np.array_equal(G, [[1.0, 0.0], [-1.0, 1.0]])


def test_design_matrix_recovers_a_known_displacement():
    net = _net(4)
    d = np.array([0.0, 1.0, 3.0, 2.5])            # truth, reference epoch 0
    G = net.design_matrix(reference=0)
    obs = np.array([d[j] - d[i] for i, j in net.pairs])
    sol, *_ = np.linalg.lstsq(G, obs, rcond=None)
    assert np.allclose(sol, d[1:])


def test_incremental_design_matrix_recovers_the_same_thing():
    net = _net(4)
    d = np.array([0.0, 1.0, 3.0, 2.5])
    G = net.incremental_design_matrix()
    obs = np.array([d[j] - d[i] for i, j in net.pairs])
    sol, *_ = np.linalg.lstsq(G, obs, rcond=None)
    assert np.allclose(np.concatenate([[0.0], np.cumsum(sol)]), d)


def test_incremental_design_matrix_handles_reversed_pairs():
    net = Network([datetime(2017, 1, 1).replace(day=1 + i) for i in range(3)],
                  [(1, 0), (1, 2)])
    d = np.array([0.0, 2.0, 5.0])
    G = net.incremental_design_matrix()
    obs = np.array([d[j] - d[i] for i, j in net.pairs])
    sol, *_ = np.linalg.lstsq(G, obs, rcond=None)
    assert np.allclose(np.concatenate([[0.0], np.cumsum(sol)]), d)
