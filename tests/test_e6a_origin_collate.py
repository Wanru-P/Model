import unittest

import numpy as np

from ppdet.data.reader import BatchCompose


class E6aOriginCollateTest(unittest.TestCase):
    def setUp(self):
        self.collate = BatchCompose([], collate_batch=False)

    def test_origin_targets_stack_without_changing_dino_gt_lists(self):
        sample = {
            'origin_gt_bbox': np.ones((2, 4), dtype=np.float32),
            'origin_gt_class': np.ones((2, 1), dtype=np.int32),
            'pad_origin_gt_mask': np.ones((2, 1), dtype=np.float32),
            'gt_bbox': np.ones((2, 4), dtype=np.float32),
            'gt_class': np.ones((2, 1), dtype=np.int32)}
        batch = self.collate([sample])

        self.assertIsInstance(batch['origin_gt_bbox'], np.ndarray)
        self.assertIsInstance(batch['origin_gt_class'], np.ndarray)
        self.assertIsInstance(batch['pad_origin_gt_mask'], np.ndarray)
        self.assertEqual(batch['origin_gt_bbox'].shape, (1, 2, 4))
        self.assertEqual(batch['origin_gt_class'].shape, (1, 2, 1))
        self.assertEqual(batch['pad_origin_gt_mask'].shape, (1, 2, 1))
        self.assertEqual(batch['origin_gt_bbox'].dtype, np.float32)
        self.assertEqual(batch['origin_gt_class'].dtype, np.int32)
        self.assertEqual(batch['pad_origin_gt_mask'].dtype, np.float32)
        self.assertIsInstance(batch['gt_bbox'], list)
        self.assertIsInstance(batch['gt_class'], list)
        self.assertEqual(batch['gt_bbox'][0].shape, (2, 4))
        self.assertEqual(batch['gt_class'][0].shape, (2, 1))

    def test_empty_origin_targets_keep_batched_rank(self):
        sample = {
            'origin_gt_bbox': np.zeros((0, 4), dtype=np.float32),
            'origin_gt_class': np.zeros((0, 1), dtype=np.int32),
            'pad_origin_gt_mask': np.zeros((0, 1), dtype=np.float32),
            'gt_bbox': np.zeros((0, 4), dtype=np.float32),
            'gt_class': np.zeros((0, 1), dtype=np.int32)}
        batch = self.collate([sample])

        self.assertEqual(batch['origin_gt_bbox'].shape, (1, 0, 4))
        self.assertEqual(batch['origin_gt_class'].shape, (1, 0, 1))
        self.assertEqual(batch['pad_origin_gt_mask'].shape, (1, 0, 1))
        self.assertIsInstance(batch['gt_bbox'], list)
        self.assertIsInstance(batch['gt_class'], list)


if __name__ == '__main__':
    unittest.main()
