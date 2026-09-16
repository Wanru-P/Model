import gc
import os
import unittest
from unittest import mock

import paddle

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight
import tools.e6a_sol_acceptance as acceptance
from tools.e6a_sol_acceptance import (
    E1_CONFIG, E6_CONFIG, assert_aux_construction, build_trainer)


class ArchitectureOnlyTrainer(object):
    def __init__(self, cfg, mode):
        self.model = create(cfg.architecture)
        self.model.train() if mode == 'train' else self.model.eval()


class E6aConfigIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')

    def test_e1_e6_e1_sequence_does_not_leak_aux_heads(self):
        with mock.patch.object(acceptance, 'Trainer', ArchitectureOnlyTrainer):
            e1_eval = build_trainer(E1_CONFIG, 'eval', aux_enabled=False)
            assert_aux_construction(e1_eval.model, False, 'first E1 eval')
            del e1_eval
            gc.collect()

            e6_eval = build_trainer(E6_CONFIG, 'eval', aux_enabled=True)
            assert_aux_construction(e6_eval.model, True, 'E6a eval')
            del e6_eval
            gc.collect()

            e1_train = build_trainer(E1_CONFIG, 'train', aux_enabled=False)
            assert_aux_construction(e1_train.model, False, 'second E1 train')
            checkpoint = os.environ.get('E6A_E1_CHECKPOINT')
            if checkpoint:
                load_weight(e1_train.model, checkpoint)
            del e1_train
            gc.collect()

            e6_disabled = build_trainer(
                E6_CONFIG, 'train', aux_enabled=False)
            assert_aux_construction(
                e6_disabled.model, False, 'E6a aux-disabled train')
            if checkpoint:
                load_weight(e6_disabled.model, checkpoint)

    def test_e1_rejects_aux_enable_request(self):
        with mock.patch.object(acceptance, 'Trainer', ArchitectureOnlyTrainer):
            with self.assertRaises(AssertionError):
                build_trainer(E1_CONFIG, 'eval', aux_enabled=True)


if __name__ == '__main__':
    unittest.main()
