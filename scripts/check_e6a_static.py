#!/usr/bin/env python3
"""Static, dependency-free provenance and E6a contract check.

Run this before a CUDA validation run. It intentionally does not import Paddle.
"""
from __future__ import annotations

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
        '4CED1D0A06EE48B29E0A5B5FE057320EAA962809078AE0076636BBEECF978C36',
}


def require(path: Path, text: str) -> None:
    content = path.read_text(encoding='utf-8')
    if text not in content:
        raise AssertionError(f'{path.relative_to(ROOT)} lacks required text: {text}')


def main() -> None:
    for rel, expected in EXPECTED.items():
        path = ROOT / rel
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest().upper()
        assert actual == expected, f'{rel} is not a verbatim upstream copy: {actual}'
        assert b'Licensed under the Apache License, Version 2.0' in content
        ast.parse(content, filename=str(path))
        print(f'PASS upstream hash/header/syntax: {rel}')

    require(ROOT / 'ppdet/data/transform/operators.py', 'retain_origin_box')
    require(ROOT / 'ppdet/data/transform/batch_operators.py', 'only_origin_box')
    require(ROOT / 'ppdet/modeling/architectures/damsdet.py', 'aux_o2m_enabled')
    reader = ROOT / 'configs/damsdet/_base_/damsdet_e6a_reader_960_o2m.yml'
    config = ROOT / 'configs/damsdet/damsdet_r50vd_aic2026_e6a_rgbir_960_dual_o2m.yml'
    require(reader, 'Multi_NormalizeBox: {retain_origin_box: true}')
    require(reader, 'PadGT: {only_origin_box: true}')
    require(config, 'static_assigner_epoch: 30')
    require(config, 'aux_o2m_weight: 1.0')
    print('PASS E6a static contract')


if __name__ == '__main__':
    main()
