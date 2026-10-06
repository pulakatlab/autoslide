"""Tests that --score-threshold in prediction.py reaches combine_prediction_masks."""

import sys
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from autoslide.src.pipeline.model import prediction, prediction_utils


class FakeModel:
    """Stand-in for Mask R-CNN returning two instances with different scores"""

    def __call__(self, batch):
        masks = torch.zeros((2, 1, 8, 8))
        masks[0, 0, :4, :4] = 1.0
        masks[1, 0, 4:, 4:] = 1.0
        return [{'masks': masks, 'scores': torch.tensor([0.9, 0.4])}]


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / 'img.png'
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    return str(path)


def _transform(img):
    return torch.zeros((3, 8, 8))


def _run(image_path, **kwargs):
    with mock.patch.object(
            prediction_utils, 'combine_prediction_masks',
            wraps=prediction_utils.combine_prediction_masks) as spy:
        mask, _ = prediction.predict_image_from_path(
            FakeModel(), image_path, torch.device('cpu'), _transform, **kwargs)
    return mask, spy


def test_threshold_reaches_combine(image_path):
    mask, spy = _run(image_path, score_threshold=0.5)
    assert spy.call_args.args[3] == 0.5
    # Only the 0.9-score instance survives
    assert (mask > 0).sum() == 16


def test_default_threshold_unchanged(image_path):
    mask, spy = _run(image_path)
    assert spy.call_args.args[3] == 0.3
    # Both instances survive at the default
    assert (mask > 0).sum() == 32


def test_process_all_images_threads_threshold(image_path, tmp_path):
    item = {'image_path': image_path, 'mask_path': str(tmp_path / 'm.png'),
            'overlay_path': str(tmp_path / 'o.png'), 'image_name': 'img.png'}
    with mock.patch.object(prediction, 'load_model',
                           return_value=(FakeModel(), torch.device('cpu'), _transform)), \
            mock.patch.object(prediction, 'find_images_to_process',
                              return_value={'svs': [item]}), \
            mock.patch.object(prediction, 'save_svs_timing_to_tracking_json'), \
            mock.patch.object(prediction, 'predict_image_from_path',
                              return_value=(None, None)) as pred:
        prediction.process_all_images(score_threshold=0.6)
        assert pred.call_args.kwargs['score_threshold'] == 0.6
        prediction.process_all_images()
        assert pred.call_args.kwargs['score_threshold'] == 0.3


def test_parse_args_score_threshold():
    with mock.patch.object(sys, 'argv', ['prediction.py']):
        assert prediction.parse_args().score_threshold == 0.3
    with mock.patch.object(sys, 'argv', ['prediction.py', '--score-threshold', '0.45']):
        assert prediction.parse_args().score_threshold == 0.45
