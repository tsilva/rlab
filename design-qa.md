# Player icon controls design QA

## Evidence

- Source visual truth: `/var/folders/wz/x29jb7_x5rdc_5dcjr4qnhg00000gn/T/codex-clipboard-a7d8f518-c411-4c3c-9eaa-db7867d61f97.png`
- Browser-rendered implementation: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-icons/04-icon-only-controls.jpg`
- Full-view comparison: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-icons/05-icon-only-full-comparison.jpg`
- Focused controls comparison: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-icons/06-icon-only-controls-comparison.jpg`
- Source pixels: 2464 × 1482. Implementation screenshot pixels: 1265 × 712. CSS viewport: 1280 × 720 at device pixel ratio 2. The full comparison normalizes both screenshots to 712 pixels high.
- State: dark desktop playback workspace, live synthetic Mario frame, policy driver, transition history, and compact controls.

## Findings

- No actionable P0, P1, or P2 differences remain.
- Routine actions are icon-only: layouts, panels, synchronized window, playback transport, seed reset, FPS apply, driver selection, policy inspection, window control, session end, timeline return, zoom, panel drag, fullscreen, and panel options.
- Every visible icon-only button has a non-empty `title` tooltip and accessible name. All decorative SVGs are hidden from the accessibility tree.
- The playback transport is one six-button row; the driver tools are one four-button row. This removes wrapping and returns substantial vertical space to research panels.
- Iconography uses one coherent, self-hosted Tabler Icons 3.44.0 subset. Filled media glyphs make pause, play, single-step, and step-10 legible at small sizes; outline glyphs distinguish secondary actions.
- Menu commands retain text because they are less frequent and context-dependent. Inputs and section headings also retain labels. This keeps the compact toolbar from becoming cryptic.
- Typography, color tokens, panel density, charts, and the aspect-correct game surface remain unchanged. The measured game ratio is 1.06634 versus the expected 256:240 ratio of 1.06667 after fractional CSS rounding.
- There is no horizontal overflow at the tested desktop viewport.

## Comparison History

- Iteration 1 found a P2 density mismatch: the initial icon pass kept labels on playback, driver, reset, and apply actions, preserving the oversized control rail shown in the source screenshot. Those routine actions were converted to icon-only controls with tooltips and accessible names.
- Iteration 1 found a P2 state-update bug: playback/recording mode code replaced the human-control button content, which could remove its icon. The mode update now changes the button's `aria-label` and tooltip only.
- Iteration 1 found a P2 small-size recognition issue: outline media glyphs were visually weak at toolbar size. They were replaced with the official filled Tabler media variants.
- Iteration 2 full-view and focused comparisons found no remaining P0/P1/P2 issues.

## Primary Interactions Tested

- Loaded the latest packaged HTML, CSS, JavaScript, and SVG sprite through the actual web player server.
- Verified all six transport controls contain no visible text, have distinct tooltips and accessible labels, and fit on one row.
- Verified all four driver controls contain no visible text and use distinct policy, human, inspect, and current-window glyphs.
- Verified 38 visible icon-only controls have tooltips and all visible buttons have accessible names.
- Verified the game canvas remains aspect-correct and the page has no horizontal overflow.
- Checked the browser console for warnings and errors: none.
- Validated the SVG sprite as XML and the frontend JavaScript syntax.

## Implementation Checklist

- [x] Use a single proper icon pack rather than emoji, CSS drawings, or ad hoc glyphs.
- [x] Remove routine button labels wherever the icon is sufficient.
- [x] Provide hover tooltips and assistive-technology names for icon-only actions.
- [x] Keep labels for inputs, headings, and context-heavy menu commands.
- [x] Preserve the player aspect ratio and dense multi-panel workspace.
- [x] Verify package delivery, browser rendering, accessibility names, layout, and console health.

final result: passed
