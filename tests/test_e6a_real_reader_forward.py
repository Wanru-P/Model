import gc
import json
import os
import unittest
from pathlib import Path

import paddle

from ppdet.core.workspace import load_config
from ppdet.engine import Trainer
from tools.e6a_sol_acceptance import (
    E1_CONFIG, E6_CONFIG, check_e1_batch_contract,
    check_origin_batch_contract, empty_gt_check, finite, scalar, seed_all)


class E6aRealReaderForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_root = os.environ.get('E6A_DATASET_ROOT')
        annotation = os.environ.get('E6A_READER_ANNOTATION')
        if not dataset_root or not annotation:
            raise unittest.SkipTest(
                'E6A_DATASET_ROOT and E6A_READER_ANNOTATION are required.')
        cls.dataset_root = str(Path(dataset_root).resolve())
        cls.annotation = annotation
        paddle.set_device(os.environ.get('E6A_TEST_DEVICE', 'cpu'))

    def build_real_trainer(self, config_path, aux_enabled):
        seed_all()
        cfg = load_config(str(config_path))
        cfg.DAMSDet['aux_o2m_enabled'] = aux_enabled
        if not aux_enabled:
            cfg.DAMSDet['aux_o2m_head_vis'] = None
            cfg.DAMSDet['aux_o2m_head_ir'] = None
        cfg.TrainDataset['dataset_dir'] = self.dataset_root
        cfg.TrainDataset['anno_path'] = self.annotation
        cfg.TrainReader['shuffle'] = False
        cfg['worker_num'] = 0
        return Trainer(cfg, mode='train')

    def test_real_e1_contract_and_e6a_single_batch_forward(self):
        e1 = self.build_real_trainer(E1_CONFIG, False)
        e1_batch = next(iter(e1.loader))
        e1_contract = check_e1_batch_contract(e1_batch)
        print('E1_BATCH_CONTRACT=' + json.dumps(e1_contract, sort_keys=True))
        self.assertEqual(e1_contract['gt_bbox']['type'], 'list')
        self.assertEqual(e1_contract['gt_class']['type'], 'list')
        del e1, e1_batch
        gc.collect()

        e6 = self.build_real_trainer(E6_CONFIG, True)
        batch = next(iter(e6.loader))
        contract = check_origin_batch_contract(batch)
        print('E6A_ORIGIN_BATCH_CONTRACT=' + json.dumps(
            contract, sort_keys=True))
        self.assertEqual(contract['origin_gt_bbox']['shape'][0], 1)
        self.assertEqual(contract['origin_gt_class']['shape'][0], 1)
        self.assertEqual(contract['pad_origin_gt_mask']['shape'][0], 1)
        batch['epoch_id'] = 0
        e6.model.train()
        outputs = e6.model(batch)
        forward = {}
        for name in ('loss', 'aux_vis_total', 'aux_ir_total'):
            self.assertTrue(finite(outputs[name]), name)
            forward[name] = scalar(outputs[name])
        print('E6A_SINGLE_BATCH_FORWARD=' + json.dumps(
            forward, sort_keys=True))

        empty_gt = empty_gt_check(e6.model, batch)
        for epoch_result in empty_gt.values():
            for modality, losses in epoch_result.items():
                for name, value in losses.items():
                    self.assertTrue(
                        float('-inf') < value < float('inf'),
                        '{}.{}'.format(modality, name))
        self.assertEqual(set(empty_gt), {'0', '30'})
        print('E6A_EMPTY_GT=' + json.dumps(empty_gt, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
