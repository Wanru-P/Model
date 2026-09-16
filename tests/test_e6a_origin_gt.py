import unittest

import numpy as np

from ppdet.data.transform.batch_operators import PadOriginGT
from ppdet.data.transform.operators import (
    BboxXYXY2XYWH, Multi_NormalizeBox, Multi_PreserveOriginGT)


class E6aOriginGTPipelineTest(unittest.TestCase):
    def test_pixel_xyxy_and_dino_cxcywh_are_independent(self):
        original = np.array(
            [[10., 20., 110., 220.], [400., 300., 430., 350.]],
            dtype=np.float32)
        sample = {
            'vis_image': np.zeros((960, 960, 3), dtype=np.uint8),
            'gt_bbox': original.copy(),
            'gt_class': np.array([[0], [10]], dtype=np.int32)}
        sample = Multi_PreserveOriginGT()(sample)
        sample = Multi_NormalizeBox()(sample)
        sample = BboxXYXY2XYWH()(sample)
        padded = PadOriginGT()([sample])[0]

        np.testing.assert_array_equal(padded['origin_gt_bbox'], original)
        self.assertEqual(padded['origin_gt_bbox'].dtype, np.float32)
        self.assertEqual(padded['origin_gt_class'].dtype, np.int32)
        self.assertTrue(np.all(padded['pad_origin_gt_mask'] == 1.))

        cxcywh = padded['gt_bbox']
        rebuilt = np.stack([
            (cxcywh[:, 0] - cxcywh[:, 2] / 2.) * 960,
            (cxcywh[:, 1] - cxcywh[:, 3] / 2.) * 960,
            (cxcywh[:, 0] + cxcywh[:, 2] / 2.) * 960,
            (cxcywh[:, 1] + cxcywh[:, 3] / 2.) * 960], axis=1)
        np.testing.assert_allclose(rebuilt, original, atol=1e-2, rtol=0.)

    def test_empty_origin_gt_is_supported(self):
        sample = {
            'origin_gt_bbox': np.zeros((0, 4), dtype=np.float32),
            'origin_gt_class': np.zeros((0, 1), dtype=np.int32)}
        padded = PadOriginGT()([sample])[0]
        self.assertEqual(padded['origin_gt_bbox'].shape, (0, 4))
        self.assertEqual(padded['origin_gt_class'].shape, (0, 1))
        self.assertEqual(padded['pad_origin_gt_mask'].shape, (0, 1))


if __name__ == '__main__':
    unittest.main()
