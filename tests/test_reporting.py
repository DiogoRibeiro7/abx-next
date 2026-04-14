"""Tests for report summary rendering."""

from __future__ import annotations

import pathlib
import sys

sys.path.append("src")

from experimetrics.reporting.summary import render_ab_report


def test_render_ab_report_markdown():
    stats = {
        "srm": {"pvalue": 0.012345},
        "primary": {
            "name": "Conversion Rate",
            "estimate": 0.1234,
            "ci_low": 0.1201,
            "ci_high": 0.1267,
            "pvalue": 0.0456,
        },
        "sensitivity": [
            {"name": "Strict", "diff": 0.0102, "ci_low": 0.005, "ci_high": 0.015},
            {"name": "Relaxed", "diff": 0.008, "ci_low": 0.003, "ci_high": 0.013},
        ],
    }
    guardrails = {
        "Latency": {"status": "pass", "pvalue": 0.8123},
        "Crash Rate": {"status": "watch"},
    }
    plots = {"Lift Chart": "![Lift](lift.png)"}

    output = render_ab_report(stats, guardrails, plots=plots)
    expected = pathlib.Path("tests/fixtures/expected_report.md").read_text()
    assert output == expected


def test_render_ab_report_html():
    markdown = render_ab_report({}, {}, format="markdown")
    html = render_ab_report({}, {}, format="html")
    assert "<h1>" in html
    assert markdown.startswith("#")
