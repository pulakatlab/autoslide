"""
Tests for the color_qc pre-flight screening module
(autoslide.src.utils.color_qc).

Note: the pre-existing tests/test_training.py, tests/test_annotation.py,
and tests/test_color_processor.py currently fail to even collect on this
branch (stale `from src import ...` / `autoslide.utils...` import paths
predating the `autoslide/src/` restructure, plus one unrelated bug in
get_section_details.py) -- this file intentionally uses the import path
confirmed to actually work in this environment
(`autoslide.src.utils.color_qc...`), not either of those stale patterns.
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from autoslide.src.utils.color_qc.features import (
    FEATURE_NAMES,
    extract_color_features,
    aggregate_slide_features,
)
from autoslide.src.utils.color_qc.reference_model import ReferenceColorModel
from autoslide.src.utils.color_qc.screen import (
    group_images_by_slide,
    screen_slide,
    screen_batch,
    format_report,
)


def _solid_hsv_image(hue_deg, saturation=180, value=180, size=64):
    """A solid-color synthetic image at a given hue (degrees), returned as
    an RGB uint8 array -- matches what extract_color_features expects for
    an array input."""
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 0] = int(hue_deg / 2)  # OpenCV hue is 0-180
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class TestExtractColorFeatures:
    def test_collagen_hue_gives_high_log_ratio(self):
        # ~250 degrees sits inside the collagen (blue) band.
        f = extract_color_features(_solid_hsv_image(250))
        assert f['log_blue_red_ratio'] > 1.0
        assert f['tissue_fraction'] == pytest.approx(1.0)

    def test_muscle_hue_gives_low_log_ratio(self):
        # ~5 degrees sits inside the muscle (red) band.
        f = extract_color_features(_solid_hsv_image(5))
        assert f['log_blue_red_ratio'] < -1.0

    def test_background_only_image_returns_nan(self):
        # Low-saturation "glass" pixels -- no tissue detected.
        img = _solid_hsv_image(0, saturation=2, value=240)
        f = extract_color_features(img)
        assert np.isnan(f['log_blue_red_ratio'])
        assert f['tissue_fraction'] == pytest.approx(0.0)

    def test_all_feature_names_present(self):
        f = extract_color_features(_solid_hsv_image(120))
        for name in FEATURE_NAMES:
            assert name in f
        assert 'tissue_fraction' in f
        assert 'other_hue_fraction' in f

    def test_accepts_a_file_path(self):
        img = _solid_hsv_image(250)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.png')
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            f = extract_color_features(path)
        assert not np.isnan(f['log_blue_red_ratio'])


class TestAggregateSlideFeatures:
    def test_mean_and_within_slide_std(self):
        per_image = [extract_color_features(_solid_hsv_image(h))
                     for h in (248, 250, 252)]
        agg = aggregate_slide_features(per_image)
        assert agg['n_images'] == 3
        assert agg['n_images_used'] == 3
        for name in FEATURE_NAMES:
            assert f'{name}_within_slide_std' in agg

    def test_nan_images_excluded_but_counted(self):
        per_image = [
            extract_color_features(_solid_hsv_image(250)),
            extract_color_features(_solid_hsv_image(0, saturation=2)),
        ]
        agg = aggregate_slide_features(per_image)
        assert agg['n_images'] == 2
        assert agg['n_images_used'] == 1

    def test_all_nan_raises(self):
        per_image = [extract_color_features(
            _solid_hsv_image(0, saturation=2))]
        with pytest.raises(ValueError):
            aggregate_slide_features(per_image)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_slide_features([])


class TestReferenceColorModel:
    def _fit_tight_reference(self, rng, n=10):
        base = {name: 50.0 + i for i, name in enumerate(FEATURE_NAMES)}
        slides = [{k: v + rng.normal(0, 1.0) for k, v in base.items()}
                  for _ in range(n)]
        return ReferenceColorModel().fit(slides), base

    def test_fit_requires_at_least_three_slides(self):
        with pytest.raises(ValueError):
            ReferenceColorModel().fit(
                [{name: 1.0 for name in FEATURE_NAMES}] * 2)

    def test_outlier_scores_much_higher_than_inlier(self):
        rng = np.random.default_rng(0)
        model, base = self._fit_tight_reference(rng)
        inlier = {k: v + rng.normal(0, 1.0) for k, v in base.items()}
        outlier = {k: v + 30.0 for k, v in base.items()}
        assert model.mahalanobis(outlier) > model.mahalanobis(inlier) * 5

    def test_verdict_tiers_respect_thresholds(self):
        rng = np.random.default_rng(1)
        model, base = self._fit_tight_reference(rng)
        inlier = {k: v + rng.normal(0, 1.0) for k, v in base.items()}
        d2_inlier = model.mahalanobis(inlier)
        model.set_thresholds(in_bounds_max=d2_inlier * 3,
                             elevated_max=d2_inlier * 10)
        assert model.score(inlier)['verdict'] == 'in_bounds'

        far = {k: v + 100.0 for k, v in base.items()}
        assert model.score(far)['verdict'] == 'out_of_bounds'

    def test_set_thresholds_validates_order(self):
        rng = np.random.default_rng(2)
        model, _ = self._fit_tight_reference(rng)
        with pytest.raises(ValueError):
            model.set_thresholds(in_bounds_max=10, elevated_max=5)

    def test_save_load_round_trip(self):
        rng = np.random.default_rng(3)
        model, base = self._fit_tight_reference(rng)
        model.set_thresholds(in_bounds_max=10, elevated_max=100)
        probe = {k: v + 5.0 for k, v in base.items()}
        d2_before = model.mahalanobis(probe)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'nested' / 'reference_model.json'
            model.save(path)
            assert path.exists()
            loaded = ReferenceColorModel.load(path)

        assert loaded.mahalanobis(probe) == pytest.approx(d2_before)
        assert loaded.thresholds_ == model.thresholds_
        assert loaded.n_slides_fit_ == model.n_slides_fit_


class TestGroupImagesBySlide:
    def _write_png(self, path, hue=250):
        img = _solid_hsv_image(hue)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def test_flat_directory_is_one_slide(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for i in range(3):
                self._write_png(td / f'roi_{i}.png')
            groups = group_images_by_slide(td)
        assert list(groups.keys()) == [td.name]
        assert len(groups[td.name]) == 3

    def test_suggested_regions_style_layout(self):
        """slide_dir/images/*.png -- and a stray top-level PNG that must
        NOT shadow the real images/ subdirectory (regression test for the
        bug found while smoke-testing against real suggested_regions data)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._write_png(td / 'SLIDE_A_visualization.png')
            for i in range(4):
                self._write_png(td / 'images' / f'roi_{i}.png')
            groups = group_images_by_slide(td)
        assert list(groups.keys()) == [td.name]
        assert len(groups[td.name]) == 4

    def test_multi_slide_directory(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for slide in ('SLIDE_A', 'SLIDE_B'):
                for i in range(2):
                    self._write_png(td / slide / 'images' / f'roi_{i}.png')
            groups = group_images_by_slide(td)
        assert set(groups.keys()) == {'SLIDE_A', 'SLIDE_B'}
        assert all(len(v) == 2 for v in groups.values())

    def test_no_images_raises_in_screen_batch(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError):
                screen_batch(td)


class TestScreenBatch:
    def _make_batch(self, td, hue, n_slides=2, n_images=3):
        td = Path(td)
        for s in range(n_slides):
            for i in range(n_images):
                img = _solid_hsv_image(hue)
                path = td / f'SLIDE_{s}' / 'images' / f'roi_{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return td

    def _tiny_reference_model_path(self, td, hue=250):
        rng = np.random.default_rng(42)
        base = extract_color_features(_solid_hsv_image(hue))
        base = {k: base[k] for k in FEATURE_NAMES}
        slides = [{k: v + rng.normal(0, 0.5) for k, v in base.items()}
                  for _ in range(10)]
        model = ReferenceColorModel().fit(slides)
        model.set_thresholds(in_bounds_max=20, elevated_max=100)
        path = Path(td) / 'ref.json'
        model.save(path)
        return path

    def test_screen_slide_matches_manual_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            ref_path = self._tiny_reference_model_path(td)
            model = ReferenceColorModel.load(ref_path)
            batch_dir = self._make_batch(td, hue=250, n_slides=1)
            image_paths = list((batch_dir / 'SLIDE_0' /
                                'images').glob('*.png'))
            result = screen_slide(image_paths, model)
        assert result['verdict'] in ('in_bounds', 'elevated', 'out_of_bounds')
        assert result['n_images'] == len(image_paths)

    def test_in_distribution_batch_passes(self):
        with tempfile.TemporaryDirectory() as td:
            ref_path = self._tiny_reference_model_path(td, hue=250)
            batch_dir = self._make_batch(td, hue=250)
            report = screen_batch(batch_dir, reference_model_path=ref_path)
        assert report['batch_overall']['verdict'] == 'in_bounds'
        assert report['n_slides'] == 2
        text = format_report(report)
        assert 'consistent with the training data' in text

    def test_shifted_batch_is_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            ref_path = self._tiny_reference_model_path(td, hue=250)
            # A muscle-red batch scored against a collagen-blue reference.
            batch_dir = self._make_batch(td, hue=5)
            report = screen_batch(batch_dir, reference_model_path=ref_path)
        assert report['batch_overall']['verdict'] == 'out_of_bounds'
        text = format_report(report)
        assert 'do not trust predictions' in text


class TestCLI:
    def test_screen_command_exit_code(self, capsys):
        from autoslide.src.utils.color_qc import cli as color_qc_cli
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rng = np.random.default_rng(7)
            base = {name: 50.0 + i for i, name in enumerate(FEATURE_NAMES)}
            slides = [{k: v + rng.normal(0, 1.0) for k, v in base.items()}
                      for _ in range(10)]
            model = ReferenceColorModel().fit(slides)
            model.set_thresholds(in_bounds_max=10, elevated_max=50)
            ref_path = td / 'ref.json'
            model.save(ref_path)

            for i in range(3):
                img = _solid_hsv_image(250)
                path = td / 'batch' / f'roi_{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            json_out = td / 'report.json'
            argv = [
                'color_qc', 'screen',
                '--input-dir', str(td / 'batch'),
                '--reference-model', str(ref_path),
                '--json-output', str(json_out),
            ]
            import sys
            old_argv = sys.argv
            sys.argv = argv
            try:
                exit_code = color_qc_cli.main()
            finally:
                sys.argv = old_argv

            assert exit_code in (0, 1)
            assert json_out.exists()
            with open(json_out) as f:
                report = json.load(f)
            assert 'batch_overall' in report

        captured = capsys.readouterr()
        assert 'COLOR/HUE PRE-FLIGHT SCREEN' in captured.out
