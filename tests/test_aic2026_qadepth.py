import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.aic2026_submit import (build_txt_files, class_aware_nms,
                                  load_class_thresholds, validate)


class SubmissionFormatTest(unittest.TestCase):

    def test_class_aware_nms_keeps_overlapping_different_classes(self):
        boxes = np.array([
            [1, 0.9, 10, 10, 30, 30],
            [1, 0.8, 11, 11, 29, 29],
            [2, 0.7, 11, 11, 29, 29],
        ], dtype=np.float32)
        kept = class_aware_nms(boxes, 0.5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[:, 0].astype(int).tolist(), [1, 2])

    def test_class_threshold_json_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'thresholds.json'
            path.write_text('[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, '
                            '0.1, 0.2, 0.3, 0.4, 0.5, 0.6]')
            thresholds = load_class_thresholds(path)
            self.assertEqual(thresholds[0], 0.1)
            self.assertEqual(thresholds[11], 0.6)

    def test_writer_normalizes_and_limits_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / 'example.png'
            Image.new('RGB', (200, 100)).save(image_path)
            txt_dir = root / 'predictions'
            txt_dir.mkdir()
            results = [{
                'bbox':
                np.array([
                    [6, 0.9, 20, 10, 120, 60],
                    [0, 0.01, 0, 0, 10, 10],
                ],
                         dtype=np.float32),
                'bbox_num':
                np.array([2]),
                'im_id':
                np.array([[0]]),
            }]
            build_txt_files(results, [image_path], txt_dir, 0.05, 100)
            validate(txt_dir, ['example'], 100)
            values = (txt_dir / 'example.txt').read_text().split()
            self.assertEqual(values[0], '6')
            self.assertAlmostEqual(float(values[1]), 0.35)
            self.assertAlmostEqual(float(values[2]), 0.35)
            self.assertAlmostEqual(float(values[3]), 0.5)
            self.assertAlmostEqual(float(values[4]), 0.5)


@unittest.skipUnless(
    importlib.util.find_spec('paddle') and importlib.util.find_spec('cv2'),
    'Paddle and OpenCV are required for transform tests')
class TriModalTransformTest(unittest.TestCase):

    def test_all_modalities_are_normalized_once(self):
        from ppdet.data.transform.operators import Multi_NormalizeImage

        sample = {
            'vis_image': np.full((2, 2, 3), 255, dtype=np.uint8),
            'ir_image': np.full((2, 2, 3), 128, dtype=np.uint8),
            'depth_image': np.full((2, 2, 1), 0.75, dtype=np.float32),
            'depth_valid_mask': np.ones((2, 2, 1), dtype=np.float32),
        }
        result = Multi_NormalizeImage(
            mean=[0., 0., 0.], std=[1., 1., 1.], norm_type='none').apply(sample)
        np.testing.assert_allclose(result['vis_image'], 1.0)
        np.testing.assert_allclose(result['ir_image'], 128.0 / 255.0)
        np.testing.assert_allclose(result['depth_image'], 0.75)
        np.testing.assert_allclose(result['depth_valid_mask'], 1.0)

    def test_depth_invalidation_clears_values_and_mask_only(self):
        from ppdet.data.transform.operators import Multi_RandomDepthInvalidation

        sample = {
            'vis_image': np.ones((20, 20, 3), dtype=np.uint8),
            'ir_image': np.ones((20, 20, 3), dtype=np.uint8),
            'depth_image': np.ones((20, 20, 1), dtype=np.float32),
            'depth_valid_mask': np.ones((20, 20, 1), dtype=np.float32),
        }
        result = Multi_RandomDepthInvalidation(
            prob=1.0, num_holes=(1, 1), area_ratio=(0.25, 0.25),
            aspect_ratio=(1.0, 1.0)).apply(sample)
        self.assertEqual(float(result['depth_image'].sum()), 300.0)
        np.testing.assert_array_equal(result['depth_image'],
                                      result['depth_valid_mask'])
        np.testing.assert_array_equal(result['vis_image'], 1)
        np.testing.assert_array_equal(result['ir_image'], 1)

    def test_modality_dropout_drops_only_selected_depth_stream(self):
        from ppdet.data.transform.operators import Multi_RandomModalityDropout

        sample = {
            'vis_image': np.ones((4, 4, 3), dtype=np.uint8),
            'ir_image': np.ones((4, 4, 3), dtype=np.uint8),
            'depth_image': np.ones((4, 4, 1), dtype=np.float32),
            'depth_valid_mask': np.ones((4, 4, 1), dtype=np.float32),
        }
        result = Multi_RandomModalityDropout(
            prob=1.0, modality_weights=(0.0, 0.0, 1.0)).apply(sample)
        np.testing.assert_array_equal(result['depth_image'], 0.0)
        np.testing.assert_array_equal(result['depth_valid_mask'], 0.0)
        np.testing.assert_array_equal(result['vis_image'], 1)
        np.testing.assert_array_equal(result['ir_image'], 1)

    def test_depth_resize_does_not_blend_invalid_zero_into_valid_depth(self):
        from ppdet.data.transform.operators import Multi_Resize

        depth = np.array([[1.0, 0.0], [1.0, 0.0]],
                         dtype=np.float32)[..., None]
        valid = (depth > 0).astype(np.float32)
        resized_depth, resized_valid = Multi_Resize.apply_depth(
            depth, valid, (4, 4))
        np.testing.assert_allclose(resized_depth[resized_valid > 0], 1.0)
        np.testing.assert_array_equal(resized_depth[resized_valid == 0], 0.0)

    def test_letterbox_keeps_aspect_ratio_and_aligns_all_modalities(self):
        from ppdet.data.transform.operators import Multi_Pad_IRVIS, Multi_Resize

        sample = {
            'vis_image': np.ones((108, 192, 3), dtype=np.uint8),
            'ir_image': np.ones((108, 192, 3), dtype=np.uint8),
            'depth_image': np.ones((108, 192, 1), dtype=np.float32),
            'depth_valid_mask': np.ones((108, 192, 1), dtype=np.float32),
            'gt_bbox': np.array([[0, 0, 192, 108]], dtype=np.float32),
        }
        sample = Multi_Resize([96, 96], keep_ratio=True).apply(sample)
        self.assertEqual(sample['vis_image'].shape[:2], (54, 96))
        sample = Multi_Pad_IRVIS(
            size=[96, 96], pad_mode=1, return_pad_mask=True).apply(sample)
        self.assertEqual(sample['vis_image'].shape, (96, 96, 3))
        self.assertEqual(sample['depth_image'].shape, (96, 96, 1))
        self.assertEqual(float(sample['pad_mask'].sum()), 54 * 96)
        np.testing.assert_allclose(sample['gt_bbox'], [[0, 21, 96, 75]])

    def test_batch_letterbox_preserves_e0_random_resize_contract(self):
        from ppdet.data.transform.batch_operators import Multi_BatchRandomResize

        sample = {
            'vis_image': np.ones((108, 192, 3), dtype=np.uint8),
            'ir_image': np.ones((108, 192, 3), dtype=np.uint8),
            'depth_image': np.ones((108, 192, 1), dtype=np.float32),
            'depth_valid_mask': np.ones((108, 192, 1), dtype=np.float32),
            'gt_bbox': np.array([[19.2, 10.8, 96.0, 54.0]], dtype=np.float32),
        }
        op = Multi_BatchRandomResize(
            target_size=96,
            keep_ratio=True,
            random_size=False,
            random_interp=False,
            pad_to_target=True,
            pad_mode=0,
            return_pad_mask=True)
        result = op([sample])[0]
        self.assertEqual(result['vis_image'].shape, (96, 96, 3))
        self.assertEqual(result['ir_image'].shape, (96, 96, 3))
        self.assertEqual(result['depth_image'].shape, (96, 96, 1))
        self.assertEqual(float(result['pad_mask'].sum()), 54 * 96)
        np.testing.assert_allclose(
            result['gt_bbox'], [[9.6, 5.4, 48.0, 27.0]], rtol=0, atol=1e-5)
        np.testing.assert_allclose(result['im_shape'], [54, 96])
        np.testing.assert_allclose(result['scale_factor'], [0.5, 0.5])

    def test_depth_jpg_uses_non_metric_branch(self):
        import cv2
        from ppdet.data.transform.operators import Multi_Decode

        raw = np.zeros((8, 8, 3), dtype=np.uint8)
        raw[2:6, 2:6] = 128
        ok, encoded = cv2.imencode('.jpg', raw)
        self.assertTrue(ok)
        depth, valid = Multi_Decode()._decode_depth(encoded.tobytes())
        self.assertGreater(float(valid.sum()), 0.0)
        self.assertGreater(float(depth.max()), 0.25)
        self.assertLessEqual(float(depth.max()), 1.0)

    def test_depth_png_is_decoded_as_uint16_before_normalizing(self):
        import cv2
        from ppdet.data.transform.operators import Multi_Decode

        raw = np.array([[0, 300], [1000, 20000]], dtype=np.uint16)
        ok, encoded = cv2.imencode('.png', raw)
        self.assertTrue(ok)
        depth, valid = Multi_Decode()._decode_depth(encoded.tobytes())
        np.testing.assert_array_equal(valid[..., 0], [[0, 1], [1, 1]])
        self.assertAlmostEqual(float(depth[0, 1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(depth[1, 1, 0]), 1.0, places=6)
        self.assertGreater(float(depth[1, 0, 0]), 0.0)

    def test_zero_affine_matrix_keeps_modalities_and_bbox_aligned(self):
        from ppdet.data.transform.operators import Multi_RandomAffine

        op = Multi_RandomAffine(prob=1.0)
        op._sample_matrix = lambda width, height: np.array(
            [[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        marker = np.zeros((32, 32, 3), dtype=np.uint8)
        marker[10:20, 8:18] = 255
        depth = np.zeros((32, 32, 1), dtype=np.float32)
        depth[10:20, 8:18] = 0.5
        valid = (depth > 0).astype(np.float32)
        sample = {
            'vis_image': marker.copy(),
            'ir_image': marker.copy(),
            'depth_image': depth,
            'depth_valid_mask': valid,
            'gt_bbox': np.array([[8, 10, 18, 20]], dtype=np.float32),
            'gt_class': np.array([[1]], dtype=np.int32),
        }
        result = op.apply(sample)
        np.testing.assert_allclose(result['gt_bbox'], [[12, 12, 22, 22]])
        self.assertEqual(
            np.unravel_index(result['vis_image'][..., 0].argmax(), (32, 32)),
            np.unravel_index(result['depth_image'][..., 0].argmax(), (32, 32)))


@unittest.skipUnless(importlib.util.find_spec('paddle'),
                     'Paddle is not installed in this interpreter')
class ZeroInitGatingTest(unittest.TestCase):

    def test_initial_gating_is_exact_identity(self):
        import paddle
        from ppdet.modeling.backbones.quality_aware_depth import (
            QualityAwareDepthGating)

        gate = QualityAwareDepthGating(rgb_channels=[8, 16, 32],
                                       depth_channels=[4, 8, 16],
                                       gate_hidden_channels=[4, 4, 4],
                                       zero_init_projection=True)
        vis = [
            paddle.randn([2, c, 8 // (2**i), 8 // (2**i)])
            for i, c in enumerate([8, 16, 32])
        ]
        ir = [paddle.randn(x.shape) for x in vis]
        depth = [
            paddle.randn([2, c, 8 // (2**i), 8 // (2**i)])
            for i, c in enumerate([4, 8, 16])
        ]
        mask = paddle.ones([2, 1, 64, 64])
        out_vis, out_ir = gate(vis, ir, depth, mask)
        for before, after in zip(vis, out_vis):
            np.testing.assert_array_equal(before.numpy(), after.numpy())
        for before, after in zip(ir, out_ir):
            np.testing.assert_array_equal(before.numpy(), after.numpy())


if __name__ == '__main__':
    unittest.main()
