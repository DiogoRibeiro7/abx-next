"""Experiment summary report generation."""

from __future__ import annotations

from typing import Iterable

from ..core.errors import ValidationError

__all__ = ['render_ab_report']

SECTION_SEPARATOR = '\n\n'

def render_ab_report(
    stats: dict[str, object],
    guardrails: dict[str, dict[str, object]],
    plots: dict[str, str] | None = None,
    *,
    format: str = 'markdown',
) -> str:
    """Render a lightweight experiment summary report."""
    if format not in {'markdown', 'html'}:
        raise ValidationError("format must be 'markdown' or 'html'.")

    srm = stats.get('srm', {}) if isinstance(stats, dict) else {}
    primary = stats.get('primary', {}) if isinstance(stats, dict) else {}
    sensitivity = stats.get('sensitivity', []) if isinstance(stats, dict) else []

    lines: list[str] = ['# Experiment Summary']

    lines.append('## Sample Ratio Mismatch')
    if isinstance(srm, dict) and srm:
        pvalue = srm.get('pvalue')
        if isinstance(pvalue, (int, float)):
            lines.append(f'SRM p-value: {pvalue:.4f}')
        else:
            lines.append('SRM results unavailable.')
    else:
        lines.append('SRM results unavailable.')

    lines.append('## Primary Metric')
    if isinstance(primary, dict) and primary:
        name = primary.get('name', 'Primary')
        estimate = primary.get('estimate')
        ci_low = primary.get('ci_low')
        ci_high = primary.get('ci_high')
        pvalue = primary.get('pvalue')
        lines.append(f'Metric: {name}')
        if all(isinstance(x, (int, float)) for x in (estimate, ci_low, ci_high)):
            lines.append(f'Estimate: {estimate:.4f} (CI {ci_low:.4f}, {ci_high:.4f})')
        if isinstance(pvalue, (int, float)):
            lines.append(f'p-value: {pvalue:.4f}')
    else:
        lines.append('Primary metric results unavailable.')

    lines.append('## Guardrails')
    if guardrails:
        for name, payload in guardrails.items():
            status = payload.get('status', 'unknown')
            pvalue = payload.get('pvalue')
            message = f'{name}: status={status}'
            if isinstance(pvalue, (int, float)):
                message += f', p-value={pvalue:.4f}'
            lines.append(message)
    else:
        lines.append('No guardrails provided.')

    lines.append('## Sensitivity Analysis')
    if isinstance(sensitivity, Iterable) and sensitivity:
        for entry in sensitivity:
            if not isinstance(entry, dict):
                continue
            name = entry.get('name', 'Exposure')
            diff = entry.get('diff')
            ci_low = entry.get('ci_low')
            ci_high = entry.get('ci_high')
            if all(isinstance(x, (int, float)) for x in (diff, ci_low, ci_high)):
                lines.append(f'{name}: diff={diff:.4f} (CI {ci_low:.4f}, {ci_high:.4f})')
            else:
                lines.append(f'{name}: insufficient data')
    else:
        lines.append('No sensitivity runs provided.')

    if plots:
        lines.append('## Plots')
        for title, snippet in plots.items():
            lines.append(f'{title}\n{snippet}')

    markdown = SECTION_SEPARATOR.join(lines) + '\n'

    if format == 'markdown':
        return markdown
    return _markdown_to_html(markdown)


def _markdown_to_html(markdown: str) -> str:
    """Convert a subset of Markdown to HTML without external dependencies."""
    html_lines: list[str] = []
    in_paragraph = False

    def _close_paragraph() -> None:
        nonlocal in_paragraph
        if in_paragraph:
            html_lines.append('</p>')
            in_paragraph = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            _close_paragraph()
            continue
        if line.startswith('### '):
            _close_paragraph()
            html_lines.append(f'<h3>{line[4:]}</h3>')
            continue
        if line.startswith('## '):
            _close_paragraph()
            html_lines.append(f'<h2>{line[3:]}</h2>')
            continue
        if line.startswith('# '):
            _close_paragraph()
            html_lines.append(f'<h1>{line[2:]}</h1>')
            continue
        line_html = line.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
        if not in_paragraph:
            html_lines.append(f'<p>{line_html}')
            in_paragraph = True
        else:
            html_lines.append(line_html)

    _close_paragraph()
    return ''.join(html_lines)