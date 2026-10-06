"""Tests for the union-of-annotators instance masks in parse_exported_labels."""

import numpy as np
import pytest
import torchvision.transforms as T
from PIL import Image

from autoslide.src.pipeline.label_handling.parse_exported_labels import (
    MAX_INSTANCES, build_instance_mask, instance_values)

W = H = 100


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def decode(mask):
    """Instance ids exactly as AugmentedCustDat.__getitem__ recovers them."""
    img = Image.fromarray(mask, mode='L')
    arr = (T.ToTensor()(img).numpy()[0] * 255).astype(np.uint8)
    return np.unique(arr)[1:]


def test_overlapping_polygons_from_two_labellers_are_one_instance():
    # same object number from two labellers, overlapping: old code summed them
    # into a third gray level, i.e. a spurious extra instance
    mask, k = build_instance_mask(
        [square(10, 10, 40, 40), square(20, 20, 50, 50)], W, H)
    assert k == 1
    assert len(decode(mask)) == 1


def test_disjoint_polygons_are_two_instances():
    mask, k = build_instance_mask(
        [square(5, 5, 20, 20), square(60, 60, 90, 90)], W, H)
    assert k == 2
    assert len(decode(mask)) == 2


def test_empty_is_all_zero():
    mask, k = build_instance_mask([], W, H)
    assert k == 0 and not mask.any()
    assert len(decode(mask)) == 0


def test_mask_positive_equals_union_and_all_instances_above_half():
    polys = [square(5, 5, 20, 20), square(
        18, 18, 30, 30), square(60, 60, 90, 90)]
    mask, k = build_instance_mask(polys, W, H)
    assert k == 2
    assert mask.min() == 0
    assert (mask[mask > 0] / 255 > 0.5).all()
    assert mask[10, 10] > 0 and mask[75, 75] > 0 and mask[50, 50] == 0


@pytest.mark.parametrize('k', [1, 2, 3, 9, 50, MAX_INSTANCES])
def test_instance_values_survive_tensor_roundtrip(k):
    vals = np.array(instance_values(k), dtype=np.uint8)
    mask = np.tile(np.concatenate([[0], vals]).astype(
        np.uint8), (4, 1))  # 0 = background
    assert len(decode(mask)) == k
    assert (vals / 255 > 0.5).all()


def test_too_many_instances_rejected():
    with pytest.raises(AssertionError):
        instance_values(MAX_INSTANCES + 1)
