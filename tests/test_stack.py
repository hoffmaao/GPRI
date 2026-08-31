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
