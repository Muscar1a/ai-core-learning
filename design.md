# Design System — AI Core Learning

A locked design system for AI Core Learning. Every page redesign reads this file before emitting code. Do not regenerate per page — extend or amend this file when the system needs to grow.

## Genre
editorial

## Macrostructure Family
- **Syllabus Directory (`index.html`)**: Index-First with Grouped Volumes & Progressive Disclosure.
- **Interactive Chapter Guides (`01-cnn/`, `02-rnn-lstm/`, `03-transformers/`)**: Workbench family.
  - Chapter 01 (CNN): Spatial Matrix Workbench.
  - Chapter 02 (RNN): Temporal Sequence Tape Workbench.
  - Chapter 03 (Transformers): Attention Subspace Laboratory Workbench.

## Theme — Warm Chamomile & E-Ink Parchment (Calibrated Low-Glare)
- `--color-paper`:        oklch(93.0% 0.020 78) /* warm oatmeal / chamomile parchment, diffuse, zero glare */
- `--color-paper-subtle`: oklch(90.5% 0.024 78)
- `--color-paper-hover`:  oklch(88.5% 0.028 78)
- `--color-paper-card`:   oklch(95.2% 0.016 78) /* crisp warm cream card */
- `--color-ink`:          oklch(26.0% 0.022 60) /* deep espresso bistre / softer than harsh charcoal */
- `--color-muted`:        oklch(46.0% 0.020 65) /* roasted umber */
- `--color-dim`:          oklch(62.0% 0.016 70)
- `--color-rule`:         oklch(84.0% 0.020 78) /* gentle tawny rule */
- `--color-rule-subtle`:  oklch(88.5% 0.018 78)
- `--color-rule-strong`:  oklch(74.0% 0.022 75)
- `--color-accent`:       oklch(54.0% 0.135 52) /* warm antique terracotta / persimmon */
- `--color-accent-soft`:  oklch(90.0% 0.038 52)
- `--color-focus`:        oklch(54.0% 0.135 52)
- `--color-success`:      oklch(52.0% 0.115 145) /* sage laurel */

## Typography
- **Display**: Newsreader (Google Fonts), weights: 400, 500, 600, style: normal (upright roman, no italic headers)
- **Body**: IBM Plex Sans, weights: 400, 500, 600
- **Code & Numerals**: JetBrains Mono, weights: 400, 500, 600, tabular figures
- **Display Tracking**: -0.02em to -0.025em for large display, line-height: 1.05 to 1.15
- **Measure**: 45ch–75ch for prose; responsive grid containers for interactive tables and matrix widgets

## Spacing
4-point named scale defined in `tokens.css`:
`--space-3xs` (2px) · `--space-2xs` (4px) · `--space-xs` (8px) · `--space-sm` (12px) · `--space-md` (16px) · `--space-lg` (24px) · `--space-xl` (40px) · `--space-2xl` (64px) · `--space-3xl` (96px).

## Motion
- Easings: `--ease-out`: `cubic-bezier(0.16, 1, 0.3, 1)` · `--ease-in`: `cubic-bezier(0.7, 0, 0.84, 0)` · `--ease-in-out`: `cubic-bezier(0.65, 0, 0.35, 1)`
- Target properties only: `opacity`, `transform`, `background-color`, `border-color`. Never `transition: all`.
- Reduced-motion: spatial motion collapses to opacity-only crossfade (≤ 150ms).

## Microinteractions Stance
- Silent success on checklist toggle and state changes; no celebratory toasts.
- Interactive matrix cells: subtle background lightness shift or high-contrast border ring; no aggressive `scale(1.15)` jumps.
- Focus rings appear instantly with ≥ 3:1 contrast ratio (`transition: none` on outline).

## CTA & Control Voice
- Primary CTA: solid dark ink with light text, subtle hover lift (translateY -1px) and accent hover.
- Secondary actions: hairline border, transparent fill, subtle paper-hover tint.
- Clickable affordances: `white-space: nowrap` strictly enforced.

## What Pages MUST Share
- Breadcrumb header linking back to `index.html` with syllabus context.
- Soft alabaster paper base and charcoal ink tokens.
- Typographic pairing (`Newsreader` + `IBM Plex Sans` + `JetBrains Mono`).
- Consistent hairline rules (`--rule-hair`) and card radius (`--radius-md`).

## What Pages MAY Differ On
- Workbench arrangement (2D spatial matrix views for CNN, unrolled time tapes for RNN, QKV multi-head heatmaps for Transformers).
- Subject-specific interactive control widgets and simulation runners.

## Exports

### tokens.css
See root `tokens.css` for the active CSS custom properties.
