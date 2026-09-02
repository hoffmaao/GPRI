"""Tests for gpri.focus: raw FMCW sweeps to SLCs, as GAMMA's gpri2_proc.py."""
import numpy as np
import pytest

from gpri import focus
from gpri.gamma import ParFile, read_slc

# A raw_par as the GPRI-II writes it (values from the BakerBend campaigns,
# with a short sweep so the synthetic file stays small).
RAW_PAR = """\
time_start: 2017-08-03 22:21:49.676416+00:00
geographic_coordinates:  48.8213216667  -121.9201816667  1252.2000  -17.5000
RF_center_freq: 1.72000000000e+10
RF_freq_min:    1.71001157538e+10
RF_freq_max:    1.72998842463e+10
RF_chirp_rate:  2.49710865319e+10
CHP_num_samp:   {ns}
TX_mode:        None
TX_RX_SEQ:      None
IMA_atten_dB:   22
ADC_capture_time:   {capture:.5f}
ADC_sample_rate:    6.25000e+06
STP_antenna_start:  -30.00000
STP_antenna_end:    50.00000
STP_rotation_speed: 5.00000
STP_gear_ratio:     72
TSC_acc_ramp_angle: 1.00000
TSC_acc_ramp_time:  {t_acc:.5f}
TSC_acc_ramp_step:  0.10000
TSC_rotation_speed: 5.00000
antenna_elevation:  10.00000
CHP_version: SW X2.01
TSC_version: SW V3.08
CHP_temperature:  33.800
TSC_temperature:  30.800
"""

NS = 2000          # samples per sweep (the instrument uses 50000)
DEC = 5


def write_raw(tmp_path, n_sweeps, target_bins=(), amp=8000.0, t_acc=None):
    """A raw file of ``n_sweeps`` de-ramped sweeps with point targets.

    Each target is a beat tone whose frequency puts it in range bin ``k``
    after the FFT; channel 2 carries the same targets with a phase offset so
    the two antennas can be told apart.  The scanner's acceleration ramp is
    shortened to one decimated line so a few dozen sweeps make an image.
    """
    tc = (NS + 1) / 6.25e6
    capture = n_sweeps * tc
    if t_acc is None:
        t_acc = tc * DEC
    (tmp_path / "x.raw_par").write_text(
        RAW_PAR.format(ns=NS, capture=capture, t_acc=t_acc))
    n = np.arange(NS + 1, dtype=float)
    rec = np.zeros((n_sweeps, NS + 1, 2), dtype=np.int16)
    for k in target_bins:
        # bin k of an rfft of length NS is the tone with k cycles per sweep;
        # fshift moves the spectrum by half a band, so a target that should
        # land in bin k is the tone at bin NS/2 - k.
        tone = np.cos(2 * np.pi * (NS // 2 - k) * n / NS)
        rec[:, :, 0] += (amp * tone).astype(np.int16)
        rec[:, :, 1] += (amp * np.cos(2 * np.pi * (NS // 2 - k) * n / NS + 0.7)).astype(np.int16)
    rec.tofile(tmp_path / "x.raw")
    return tmp_path / "x.raw", tmp_path / "x.raw_par"


def test_rawpar_reads_gamma_keywords(tmp_path):
    raw, par = write_raw(tmp_path, 20)
    rp = focus.RawPar.load(par)
    assert rp.ns == NS
    assert rp.time_start == "2017-08-03 22:21:49.676416+00:00"
    assert rp.lat == pytest.approx(48.8213216667)
    assert rp.geoid == -17.5
    assert rp.TSC_version == "V3.08"
    assert rp.TSC_acc_ramp_time == pytest.approx((NS + 1) / 6.25e6 * DEC, abs=1e-5)


def test_geometry_matches_gpri2_proc(tmp_path):
    """The derived numbers gpri2_proc.py prints for the BakerBend setup."""
    raw, par = write_raw(tmp_path, 100)
    rp = focus.RawPar.load(par)
    rp.ns, rp.capture_time, rp.TSC_acc_ramp_time = 50000, 16.9716, 0.5858
    g = focus.FocusGeometry(rp, focus.baker_options(), nl_tot=2121)
    assert g.rps == pytest.approx(0.750349, abs=1e-6)
    assert g.ns_min == 400 and g.ns_max == 22500 and g.ns_out == 22101
    assert g.nl_tot_dec == 424 and g.nl_acc == 14 and g.nl_image == 396
    assert g.tcycle * DEC == pytest.approx(4.000080e-02, rel=1e-6)
    assert g.sqfc == pytest.approx(1.044533, abs=1e-5)
    # squint: about +-2 lines (0.4 deg) across the band, zero at band centre
    assert abs(g.sq_lin[(g.nsamp - 1) // 2]) < 1e-12
    assert 1.5 < abs(g.sq_lin[0]) < 2.5 and np.sign(g.sq_lin[0]) != np.sign(g.sq_lin[-1])


def test_slc_par_is_gamma_format(tmp_path):
    raw, par = write_raw(tmp_path, 100)
    rp = focus.RawPar.load(par)
    rp.ns, rp.capture_time, rp.TSC_acc_ramp_time = 50000, 16.9716, 0.5858
    g = focus.FocusGeometry(rp, focus.baker_options(), nl_tot=2121)
    text = g.slc_par(1)
    p = ParFile.loads(text)
    assert p.header.startswith("Gamma Interferometric SAR Processor")
    assert "CH1 lower" in text and "CH2 upper" in g.slc_par(2)
    assert p.shape == (396, 22101)
    assert p.float("start_time") == pytest.approx(80510.256428, abs=1e-5)
    assert p.float("end_time") == pytest.approx(80526.056744, abs=1e-5)
    assert p.float("near_range_slc") == pytest.approx(300.139581, abs=1e-5)
    assert p.float("GPRI_az_start_angle") == pytest.approx(-27.955467, abs=1e-5)
    assert p.float("GPRI_az_angle_step") == pytest.approx(0.200004, abs=1e-6)
    assert p.float("prf") == pytest.approx(24.9995, abs=1e-4)
    assert p.float("receiver_gain") == 38.0
    assert p.floats("GPRI_rx2_coord")[2] - p.floats("GPRI_rx1_coord")[2] == pytest.approx(0.25)
    assert p.float("radar_frequency") == 1.72e10


def test_focus_puts_point_targets_in_their_range_bins(tmp_path):
    n_sweeps = 60
    raw, par = write_raw(tmp_path, n_sweeps, target_bins=(300, 700))
    opts = focus.FocusOptions(dec=DEC, zero=20, rmin=0.0, kbeta=3.84)
    g = focus.focus(raw, par, tmp_path / "xl.slc", tmp_path / "xu.slc", opts)
    assert g.nl_tot == n_sweeps
    lower = read_slc(tmp_path / "xl.slc", tmp_path / "xl.slc.par")
    upper = read_slc(tmp_path / "xu.slc", tmp_path / "xu.slc.par")
    assert lower.shape == g.shape == (g.nl_image, g.ns_out)
    assert lower.dtype == np.complex64
    prof = np.abs(lower).mean(axis=0)
    assert prof[:500].argmax() == 300 and 500 + prof[500:].argmax() == 700
    assert prof[300] > 100 * np.median(prof) and prof[700] > 100 * np.median(prof)
    # the two channels see the same targets with the injected phase offset
    dphi = np.angle(upper[:, 300] * np.conj(lower[:, 300]))
    assert np.allclose(dphi, -0.7, atol=0.02)      # spectra are conjugated
    # far targets are weighted up by the r**1.5 range scaling
    assert prof[700] / prof[300] == pytest.approx(g.scale[700] / g.scale[300], rel=0.05)


def test_focus_writes_big_endian_like_gamma(tmp_path):
    raw, par = write_raw(tmp_path, 40, target_bins=(100,))
    opts = focus.FocusOptions(dec=DEC, zero=20, rmin=0.0)
    focus.focus(raw, par, tmp_path / "xl.slc", tmp_path / "xu.slc", opts)
    little = np.abs(np.fromfile(tmp_path / "xl.slc", dtype="<c8"))
    big = np.abs(np.fromfile(tmp_path / "xl.slc", dtype=">c8"))
    assert np.isfinite(big).all() and big.max() < 1e6
    assert np.isnan(little).any() or np.nanmax(little) > 1e6


def test_focus_campaign_writes_scene(tmp_path):
    camp = tmp_path / "campaign"
    for sub, ids in (("raw", ["20170827_234940"]), ("raw2", ["20170828_064211", "20170828_064411"])):
        d = camp / sub
        d.mkdir(parents=True)
        for i in ids:
            r, p = write_raw(d, 40, target_bins=(50,))
            r.rename(d / f"{i}.raw")
            p.rename(d / f"{i}.raw_par")
    found = focus.find_raw(camp)
    assert [focus.scene_id(r) for r, _ in found] == \
        ["20170827_234940", "20170828_064211", "20170828_064411"]
    (camp / "RAW_list").write_text("raw2/20170828_064411.raw raw2/20170828_064411.raw_par\n")
    assert len(focus.find_raw(camp)) == 3          # RAW_list is opt-in
    assert len(focus.find_raw(camp, camp / "RAW_list")) == 1

    scene = tmp_path / "scene"
    opts = focus.FocusOptions(dec=DEC, zero=20, rmin=0.0)
    done = focus.focus_campaign(camp, scene, opts, workers=1, log=lambda *a: None)
    assert done == ["20170827_234940", "20170828_064211", "20170828_064411"]
    tab = (scene / "SLCu_tab").read_text().splitlines()
    assert tab[0] == "slc/20170827_234940u.slc  slc/20170827_234940u.slc.par"
    assert (scene / "slc" / "20170828_064411l.slc.par").exists()
    # a second run skips what is there
    done2 = focus.focus_campaign(camp, scene, opts, workers=1, log=lambda *a: None)
    assert done2 == done

    from gpri.stack import SlcPairStack
    st = SlcPairStack.from_tab(scene / "SLCu_tab", lags=(1,))
    assert st.n_pairs == 2
