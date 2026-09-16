#!/usr/bin/env python3
"""Dependency-free static gate for the Sol redo."""
from __future__ import print_function

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'ppdet/modeling/heads/ppyoloe_head.py':
        '2BDB1457A71C2280267CF87D73ECBB8FCEA5559806825048B90F918B03EC0035',
    'ppdet/modeling/assigners/atss_assigner.py':
        '0F725D402C42F518D463FC0EB6CD91BF2BFD86889B9830DD73F2331A4450368B',
    'ppdet/modeling/assigners/task_aligned_assigner.py':
        '4CED1D0A06EE48B29E0A5B5FE057320EAA962809078AE0076636BBEECF978C36'}


def main():
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest().upper()
        if actual != expected:
            raise AssertionError('{} hash mismatch: {}'.format(relative, actual))
        if b'Licensed under the Apache License, Version 2.0' not in content:
            raise AssertionError('{} lost its Apache header.'.format(relative))
        ast.parse(content, filename=str(path))

    config = (ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e6a_sol_960_dual_o2m.yml').read_text(
        encoding='utf-8')
    reader = (ROOT / 'configs/damsdet/_base_/damsdet_e6a_sol_reader_960_letterbox.yml').read_text(
        encoding='utf-8')
    architecture = (ROOT / 'ppdet/modeling/architectures/damsdet.py').read_text(
        encoding='utf-8')
    required_config = (
        'aux_o2m_weight: 1.0', 'static_assigner_epoch: 30',
        'fpn_strides: [8, 16, 32]', 'topk: 9', 'topk: 13',
        'alpha: 1.0', 'beta: 6.0')
    for text in required_config:
        if text not in config:
            raise AssertionError('E6a config lacks: ' + text)
    for text in ('Multi_PreserveOriginGT: {}', 'Multi_NormalizeBox: {}',
                 'BboxXYXY2XYWH: {}', 'PadOriginGT: {}'):
        if text not in reader:
            raise AssertionError('E6a reader lacks: ' + text)
    if 'detach()' in architecture[architecture.index('if self.training and self.aux_o2m_enabled:'):]:
        raise AssertionError('Auxiliary feature path must not detach.')
    for relative in ('tools/e6a_sol_acceptance.py',
                     'scripts/check_e6a_ready.py'):
        tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = [child for child in ast.walk(node)
                          if child is not node and isinstance(
                              child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Lambda))]
                if nested:
                    raise AssertionError('{} contains nested functions.'.format(relative))
    print('E6a Sol static check passed')


if __name__ == '__main__':
    main()
