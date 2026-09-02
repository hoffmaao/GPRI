"""Pair discovery and tiled reading of a GAMMA diff0 directory."""
import numpy as np
import pytest

from gpri.gamma import write_image
from gpri.stack import DiffStack, find_pairs

PAR = """Gamma Interferometric SAR Processor (ISP) - Image Parameter File
title: test
sensor: GPRI 2.0
date:  2017 08 03
range_samples:    9
azimuth_lines:    4
image_format:          FCOMPLEX
range_pixel_spacing:   0.750349  m
near_range_slc:        300.139581  m
radar_frequency:       1.720000e+10  Hz
GPRI_az_start_angle:    -27.955467  degrees
GPRI_az_angle_step:   2.000040e-01  degrees
GPRI_ant_elev_angle:     10.000000  degrees
"""
IDS = ["20170803_222136u", "20170803_222556u", "20170803_222756u", "20170803_222956u"]
SHAPE = (4, 9)


@pytest.fixture
def scene(tmp_path):
    """A miniature BakerBend1: an SLC_tab, a daisy chain of diffs, and .cc files."""
    (tmp_path / "slc").mkdir()
    (tmp_path / "diff0").mkdir()
    rows = []
    for sid in IDS:
        (tmp_path / "slc" / f"{sid}.slc.par").write_text(PAR)
        rows.append(f"slc/{sid}.slc  slc/{sid}.slc.par")
    (tmp_path / "SLCu_tab").write_text("\n".join(rows) + "\n")

    rng = np.random.default_rng(0)
    # GAMMA emits a self-pair first, then the chain
    names = [(IDS[0], IDS[0])] + [(IDS[i], IDS[i + 1]) for i in range(len(IDS) - 1)]
    for ref, sec in names:
        base = tmp_path / "diff0" / f"{ref}_{sec}"
        write_image(str(base) + ".diff",
                    (rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)).astype(np.complex64))
        write_image(str(base) + ".cc", rng.random(SHAPE).astype(np.float32), "FLOAT")
        # decoys that must not be picked up by a ".diff" query
        write_image(str(base) + ".adf.diff", np.ones(SHAPE, np.complex64))
        (tmp_path / "diff0" / f"{ref}_{sec}.off").write_text("title: x\n")
    return tmp_path


def test_find_pairs_drops_the_self_pair(scene):
    found = find_pairs(scene / "diff0")
    assert len(found) == 3
    assert all(ref != sec for ref, sec, _ in found)


def test_find_pairs_keeps_the_self_pair_on_request(scene):
    assert len(find_pairs(scene / "diff0", exclude_self=False)) == 4


def test_find_pairs_does_not_match_adf_files(scene):
    assert all(p.name.endswith(".diff") and ".adf." not in p.name
               for _, _, p in find_pairs(scene / "diff0"))


def test_find_pairs_can_select_adf(scene):
    found = find_pairs(scene / "diff0", suffix=".adf.diff")
    assert len(found) == 3 and all(".adf.diff" in p.name for _, _, p in found)


def test_find_pairs_is_time_ordered(scene):
    found = find_pairs(scene / "diff0")
    assert [r for r, _, _ in found] == IDS[:3]


def test_stack_geometry_comes_from_the_slc_par(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    assert st.shape == SHAPE
    assert st.n_pairs == 3 and st.n_epochs == 4
    assert st.wavelength == pytest.approx(0.0174298, abs=1e-6)
    assert st.slant_range().shape == (9,)
    assert st.azimuth_angles().shape == (4,)


def test_network_pairs_are_the_daisy_chain(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    assert st.network.pairs.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert st.network.is_connected()


def test_read_pair_matches_the_file(scene):
    from gpri.gamma import read_image
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    direct = read_image(st.paths[0], shape=SHAPE, image_format="FCOMPLEX")
    assert np.allclose(st.read_pair(0), direct)


def test_read_pair_tile_matches_the_full_read(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    full = st.read_pair(0)
    assert np.allclose(st.read_pair(0, slice(1, 3), slice(2, 7)), full[1:3, 2:7])


def test_read_patch_covers_every_pair(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    ifg, cc = st.read_patch(slice(0, 4), slice(0, 9))
    assert ifg.shape == (3, 4, 9) and cc.shape == (3, 4, 9)
    assert np.all(cc >= 0)


def test_patches_tile_the_whole_scene_exactly_once(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    seen = np.zeros(SHAPE, int)
    for rows, cols, ifg, cc in st.patches(rows=3, cols=4):
        seen[rows, cols] += 1
        assert ifg.shape[0] == st.n_pairs
        assert np.allclose(ifg[0], st.read_pair(0)[rows, cols])
    assert np.all(seen == 1)


def test_patch_shape_respects_the_memory_budget(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    rows, cols = st.patch_shape(max_gib=1e-7)
    assert rows >= 1 and cols >= 1
    assert rows * cols * st.n_pairs * 12 <= max(1e-7 * 2 ** 30, cols * st.n_pairs * 12)


def test_missing_cc_falls_back_to_magnitude(scene):
    for f in (scene / "diff0").glob("*.cc"):
        f.unlink()
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    ifg, cc = st.read_patch(slice(None), slice(None))
    assert np.allclose(cc[0], np.abs(ifg[0]))


def test_empty_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="interferograms"):
        DiffStack.from_directory(tmp_path / "empty")


def test_works_without_an_slc_tab(scene):
    st = DiffStack.from_directory(scene / "diff0",
                                  par=scene / "slc" / f"{IDS[0]}.slc.par")
    assert st.n_pairs == 3 and st.n_epochs == 4


def test_repr_is_informative(scene):
    st = DiffStack.from_directory(scene / "diff0", slc_tab=scene / "SLCu_tab")
    assert "3 pairs" in repr(st) and "4 epochs" in repr(st)


# ------------------------------------------------------------- SLC-formed pairs
from gpri.stack import SlcPairStack, coherence_window                   # noqa: E402

BIG = (12, 40)
BIG_PAR = PAR.replace("range_samples:    9", "range_samples:    40") \
             .replace("azimuth_lines:    4", "azimuth_lines:    12")
IDS_L = [s[:-1] + "l" for s in IDS]


@pytest.fixture
def slc_scene(tmp_path):
    """Both antennas' SLCs on disk, a tab for the upper only, as GAMMA leaves it."""
    (tmp_path / "slc").mkdir()
    rng = np.random.default_rng(1)
    base = rng.normal(size=BIG) + 1j * rng.normal(size=BIG)   # common speckle
    rows = []
    for k, (u, lo) in enumerate(zip(IDS, IDS_L)):
        # coherent scene that walks in phase by 0.3 rad per epoch, plus noise
        for sid, noise in ((u, 0.2), (lo, 0.5)):
            s = base * np.exp(1j * 0.3 * k) + noise * (rng.normal(size=BIG)
                                                      + 1j * rng.normal(size=BIG))
            write_image(tmp_path / "slc" / f"{sid}.slc", s.astype(np.complex64))
            (tmp_path / "slc" / f"{sid}.slc.par").write_text(BIG_PAR)
        rows.append(f"slc/{u}.slc  slc/{u}.slc.par")
    (tmp_path / "SLCu_tab").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_coherence_window_normalised():
    wa, wr = coherence_window((5, 5), "triangular")
    assert wa.sum() == pytest.approx(1.0) and wr.sum() == pytest.approx(1.0)
    assert wa[2] == wa.max() and wa[0] == wa[-1]
    b, _ = coherence_window((3, 3), "boxcar")
    assert np.allclose(b, 1 / 3)
    with pytest.raises(ValueError):
        coherence_window((3, 3), "hann")


def test_slc_pairs_match_gamma_product(slc_scene):
    """Pair (i, j) is exactly s_i * conj(s_j) at one look - GAMMA's SLC_intf."""
    from gpri.gamma import read_image
    st = SlcPairStack.from_tab(slc_scene / "SLCu_tab")
    assert st.n_pairs == 3 and st.n_epochs == 4 and st.shape == BIG
    assert list(map(tuple, st.network.pairs)) == [(0, 1), (1, 2), (2, 3)]
    a = read_image(slc_scene / "slc" / f"{IDS[1]}.slc")
    b = read_image(slc_scene / "slc" / f"{IDS[2]}.slc")
    np.testing.assert_allclose(st.read_pair(1), a * np.conj(b), rtol=1e-6)
    # the tile interface and the whole-frame one agree
    np.testing.assert_array_equal(st.read_pair(1, slice(2, 5), slice(0, 7)),
                                  (a * np.conj(b))[2:5, 0:7])
    # scene phase walks +0.3 rad/epoch and pair (i,j) carries theta_i - theta_j
    assert np.angle(st.read_pair(0)[3:-3, 3:-3].mean()) == pytest.approx(-0.3, abs=0.05)


def test_slc_pairs_coherence_in_range_and_ordered(slc_scene):
    up = SlcPairStack.from_directory(slc_scene / "slc", antenna="u")
    lo = SlcPairStack.from_directory(slc_scene / "slc", antenna="l")
    assert lo.n_pairs == up.n_pairs
    cu, cl = up.read_coherence(0), lo.read_coherence(0)
    assert cu.shape == BIG and cu.min() >= 0 and cu.max() <= 1 + 1e-6
    # lower antenna was given more noise: it must come out less coherent
    assert cl.mean() < cu.mean()
    # the daisy chain is the default; higher lags add the closing pairs
    tri = SlcPairStack.from_directory(slc_scene / "slc", antenna="u", lags=(1, 2, 3))
    assert tri.n_pairs == 3 + 2 + 1
    assert list(map(tuple, tri.network.pairs))[:3] == [(0, 1), (0, 2), (0, 3)]
    from gpri.timeseries import triplets
    assert len(triplets(tri.network)) > 0
    with pytest.raises(ValueError):
        SlcPairStack.from_directory(slc_scene / "slc", lags=(0,))
    with pytest.raises(FileNotFoundError):
        SlcPairStack.from_directory(slc_scene / "slc", antenna="x")


def test_slc_pairs_cache_bounded(slc_scene):
    st = SlcPairStack.from_tab(slc_scene / "SLCu_tab", lags=(1, 2))
    for p in range(st.n_pairs):
        st.read_pair(p)
        assert len(st._slcs) <= 3          # max(lags) + 1
    st.close()
    assert not st._slcs


def test_slc_pairs_multilook_geometry(slc_scene):
    st = SlcPairStack.from_tab(slc_scene / "SLCu_tab", looks=(3, 4))
    assert st.shape == (4, 10)
    assert st.par.range_pixel_spacing == pytest.approx(0.750349 * 4)
    assert st.par.float("GPRI_az_angle_step") == pytest.approx(0.200004 * 3)
    # multilooked azimuth centre sits mid-way through the 3 lines it averages
    assert st.par.float("GPRI_az_start_angle") == pytest.approx(-27.955467 + 0.200004)
    ifg = st.read_pair(0)
    assert ifg.shape == (4, 10) and ifg.dtype == np.complex64
    one = SlcPairStack.from_tab(slc_scene / "SLCu_tab").read_pair(0)
    np.testing.assert_allclose(ifg[0, 0], one[:3, :4].mean(), rtol=1e-5)
    # one-look closure is identically zero; multilooking makes it non-zero
    from gpri.timeseries import closure_phase, triplets
    for looks, expect_zero in (((1, 1), True), ((3, 4), False)):
        s3 = SlcPairStack.from_tab(slc_scene / "SLCu_tab", lags=(1, 2), looks=looks)
        ph = np.stack([np.angle(s3.read_pair(p)) for p in range(s3.n_pairs)])
        c = closure_phase(ph, s3.network, triplets(s3.network))
        assert (np.abs(c).max() < 1e-5) == expect_zero


def test_slc_pairs_walk_patches(slc_scene):
    st = SlcPairStack.from_tab(slc_scene / "SLCu_tab")
    seen = 0
    for rs, cs, ifg, cc in st.patches(rows=5, cols=40):
        assert ifg.shape[0] == st.n_pairs and cc.shape == ifg.shape
        seen += ifg.shape[1]
    assert seen == BIG[0]
    assert "SlcPairStack(3 pairs, 4 epochs" in repr(st)


def test_cli_opens_slc_formed_stacks(slc_scene, capsys):
    """`gpri info --antenna lower` and `--lags 1 2` work without a diff0."""
    from gpri.cli import build_parser, _open
    p = build_parser()
    args = p.parse_args(["info", str(slc_scene), "--antenna", "lower"])
    st = _open(args)
    assert isinstance(st, SlcPairStack) and st.n_pairs == 3
    assert st.images[0].name.endswith("l.slc")
    args = p.parse_args(["info", str(slc_scene), "--lags", "1", "2",
                         "--looks-pairs", "3", "4", "--max-pairs", "4"])
    st = _open(args)
    assert st.n_pairs == 4 and st.shape == (4, 10)
    assert list(map(tuple, st.network.pairs)) == [(0, 1), (0, 2), (1, 2), (1, 3)]
    # no diff0 here: the default path must say so rather than crash
    import pytest as _pt
    with _pt.raises(SystemExit):
        _open(p.parse_args(["info", str(slc_scene)]))


def test_slc_pairs_crop_a_widened_sweep(slc_scene):
    """A campaign whose sweep was widened part-way keeps the common lines.

    20170827 scanned -30..50 deg for six hours and -30..60 deg after that;
    the scans start at the same angle, so the longer images are cropped at
    the end and the stack is the leading block every epoch has.
    """
    from gpri.gamma import read_image, write_image
    extra = 3
    long_id = IDS[2]
    path = slc_scene / "slc" / f"{long_id}.slc"
    short = read_image(path, shape=BIG, image_format="FCOMPLEX")
    tail = np.ones((extra, BIG[1]), dtype=np.complex64)
    write_image(path, np.vstack([short, tail]))
    par = (slc_scene / "slc" / f"{long_id}.slc.par").read_text()
    assert f"azimuth_lines:    {BIG[0]}" in par
    (slc_scene / "slc" / f"{long_id}.slc.par").write_text(
        par.replace(f"azimuth_lines:    {BIG[0]}", f"azimuth_lines:    {BIG[0] + extra}"))

    st = SlcPairStack.from_tab(slc_scene / "SLCu_tab")
    assert st.shape == BIG and st.slc_par.azimuth_lines == BIG[0]
    a = read_image(slc_scene / "slc" / f"{IDS[1]}.slc", shape=BIG, image_format="FCOMPLEX")
    np.testing.assert_allclose(st.read_pair(1), a * np.conj(short), rtol=1e-6)

    # a file that is not a whole number of lines is refused
    with open(path, "ab") as f:
        f.write(b"\0" * 4)
    with pytest.raises(ValueError, match="whole number"):
        SlcPairStack.from_tab(slc_scene / "SLCu_tab")
