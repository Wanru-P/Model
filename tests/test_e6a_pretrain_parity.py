import gc
import os
import unittest
from unittest import mock

import paddle

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_pretrain_weight
import tools.e6a_sol_acceptance as acceptance
from tools.e6a_sol_acceptance import (
    E1_CONFIG, E6_CONFIG, build_trainer, inspect_pretrain_loading, seed_all)


class ArchitectureOnlyTrainer(object):
    def __init__(self, cfg, mode):
        self.model = create(cfg.architecture)
        self.checkpoint_mode = 'multi' if mode == 'train' else 'default'

    def load_weights(self, weights):
        load_pretrain_weight(
            self.model, weights, mode=self.checkpoint_mode)


class E6aPretrainParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        checkpoint = os.environ.get('E6A_COCO_PRETRAIN')
        if not checkpoint:
            raise unittest.SkipTest('E6A_COCO_PRETRAIN is required.')
        with mock.patch.object(acceptance, 'Trainer', ArchitectureOnlyTrainer):
            seed_all()
            cls.e1 = build_trainer(
                E1_CONFIG, 'train', aux_enabled=False)
            seed_all()
            cls.e6 = build_trainer(
                E6_CONFIG, 'train', aux_enabled=True)
        vis_parameter_ids = {
            id(item) for item in
            cls.e6.model.aux_o2m_head_vis.parameters()}
        ir_parameter_ids = {
            id(item) for item in
            cls.e6.model.aux_o2m_head_ir.parameters()}
        cls.aux_independent = (
            cls.e6.model.aux_o2m_head_vis is not
            cls.e6.model.aux_o2m_head_ir and
            vis_parameter_ids.isdisjoint(ir_parameter_ids))
        cls.audit = inspect_pretrain_loading(cls.e1, cls.e6, checkpoint)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'e1'):
            del cls.e1
        if hasattr(cls, 'e6'):
            del cls.e6
        gc.collect()

    def test_e6a_main_initialization_matches_e1_before_pretrain(self):
        before = self.audit['before_pretrain_main_state']
        self.assertTrue(before['exact_equal'])
        self.assertEqual(before['max_abs_diff'], 0.0)

    def test_e6a_main_state_matches_e1_after_coco_pretrain(self):
        after = self.audit['after_pretrain_main_state']
        self.assertTrue(after['exact_equal'])
        self.assertEqual(after['max_abs_diff'], 0.0)

    def test_match_sets_and_auxiliary_initialization(self):
        self.assertTrue(self.audit['matched_main_set_equal'])
        self.assertTrue(self.audit['unmatched_main_set_equal'])
        self.assertTrue(self.audit['aux_vis_initialized_from_scratch'])
        self.assertTrue(self.audit['aux_ir_initialized_from_scratch'])
        self.assertTrue(self.aux_independent)


if __name__ == '__main__':
    unittest.main()
