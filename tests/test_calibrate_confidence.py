"""Tests for scripts.calibrate_confidence — Platt-scaling recalibration."""

__author__ = "Luca Ostinelli"

import json

from scripts.calibrate_confidence import (
    MIN_SAMPLES,
    apply_platt,
    fit_platt,
    load_calibration,
    save_calibration,
)


class TestFitPlatt:
    def test_below_min_samples_returns_none(self):
        pairs = [(0.8, 1)] * (MIN_SAMPLES - 1)
        assert fit_platt(pairs) is None

    def test_perfectly_separable_data_fits_monotone_mapping(self):
        # Confident positives labeled correct, confident negatives labeled wrong.
        pairs = [(0.9, 1)] * 20 + [(0.3, 0)] * 20
        params = fit_platt(pairs, epochs=200)
        assert params is not None
        low = apply_platt(0.3, params)
        high = apply_platt(0.9, params)
        assert high > low  # monotone: higher raw confidence -> higher calibrated confidence


class TestApplyPlatt:
    def test_none_params_is_noop(self):
        assert apply_platt(0.73, None) == 0.73

    def test_identity_params_roughly_preserve_value(self):
        assert abs(apply_platt(0.6, (1.0, 0.0)) - 0.6) < 1e-6


class TestCalibrationRoundtrip:
    def test_save_and_load_fitted(self, tmp_path, monkeypatch):
        cal_path = tmp_path / "calibration.json"
        monkeypatch.setattr("scripts.calibrate_confidence.CALIBRATION_JSON", cal_path)
        save_calibration((1.2, -0.1), n_samples=50)
        params = load_calibration()
        assert params is not None
        assert params[0] == 1.2
        assert params[1] == -0.1

    def test_save_and_load_unfitted(self, tmp_path, monkeypatch):
        cal_path = tmp_path / "calibration.json"
        monkeypatch.setattr("scripts.calibrate_confidence.CALIBRATION_JSON", cal_path)
        save_calibration(None, n_samples=5)
        assert load_calibration() is None
        payload = json.loads(cal_path.read_text())
        assert payload["fitted"] is False

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.calibrate_confidence.CALIBRATION_JSON", tmp_path / "nope.json")
        assert load_calibration() is None
