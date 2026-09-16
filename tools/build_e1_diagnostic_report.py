#!/usr/bin/env python3
"""Combine the two read-only E1 diagnostic reports into one handoff file."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--padding-dir', type=Path,
                        default=Path('output/diagnostics/e1_padding'))
    parser.add_argument('--submission-dir', type=Path,
                        default=Path('output/diagnostics/e1_submission'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/diagnostics/E1_DIAGNOSTIC_REPORT.md'))
    return parser.parse_args()


def read_report(directory):
    report = directory / 'report.md'
    if not report.is_file():
        raise FileNotFoundError('Missing diagnostic report: {}'.format(report))
    return report.read_text(encoding='utf-8')


def main():
    args = parse_args()
    padding = read_report(args.padding_dir)
    submission = read_report(args.submission_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = ('# E1_DIAGNOSTIC_REPORT\n\n'
               'This report combines two read-only, full-400-validation '
               'diagnostics using the same E1 checkpoint. No production code '
               'or training configuration was changed.\n\n'
               '---\n\n' + padding + '\n---\n\n' + submission)
    args.output.write_text(content, encoding='utf-8')
    print('Combined diagnostic report: {}'.format(args.output.resolve()))


if __name__ == '__main__':
    main()
