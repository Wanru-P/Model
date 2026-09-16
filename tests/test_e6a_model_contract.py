import unittest
from pathlib import Path

from ppdet.core.workspace import create, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e6a_sol_960_dual_o2m.yml'


class E6aModelContractTest(unittest.TestCase):
    def test_dual_heads_are_independent_and_follow_fixed_recipe(self):
        cfg = load_config(str(CONFIG))
        model = create(cfg.architecture)
        vis = model.aux_o2m_head_vis
        ir = model.aux_o2m_head_ir
        self.assertIsNot(vis, ir)
        self.assertEqual(vis.in_channels, [256, 256, 256])
        self.assertEqual(ir.in_channels, [256, 256, 256])
        self.assertEqual(list(vis.fpn_strides), [8, 16, 32])
        self.assertEqual(list(ir.fpn_strides), [8, 16, 32])
        self.assertEqual(vis.static_assigner_epoch, 30)
        self.assertEqual(ir.static_assigner_epoch, 30)
        self.assertEqual(model.aux_o2m_weight, 1.0)


if __name__ == '__main__':
    unittest.main()
