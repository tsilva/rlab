# Player workspace design QA

**Evidence**

- Source visual truth: `/Users/tsilva/.codex/generated_images/019f8a2e-1a99-76d0-8b3a-217f3c39a914/exec-dfc4c8c9-a3cc-4bbd-a4d0-a00665f3cd87.png`
- Browser-rendered implementation: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-audit/06-composable-workspace-final.png`
- Synchronized secondary window: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-audit/07-synchronized-observation-window-final.png`
- Full-view comparison: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-audit/08-reference-comparison.png`
- Focused controls comparison: `/Users/tsilva/.codex/visualizations/2026/07/22/019f8a2e-1a99-76d0-8b3a-217f3c39a914/rlab-player-audit/09-controls-comparison.png`
- Source pixels: 1487 × 1058. Implementation screenshot pixels: 1265 × 791. CSS viewport: 1280 × 800 at device pixel ratio 1. The full-view comparison normalizes both images to 791 pixels high; their widths remain proportional because the reference canvas is taller than the tested browser viewport.
- State: dark desktop playback workspace, live Mario frame, policy driver, retained transition history, and the observation panel assigned to a second synchronized window.

**Findings**

- No actionable P0, P1, or P2 differences remain.
- Fonts and typography: the system Inter-style stack, compact monospaced telemetry, weights, hierarchy, and wrapping preserve the reference's dense research-tool character. Small labels remain readable in the narrow control rail.
- Spacing and layout rhythm: the compact header, large game-first 12-column workspace, slim controls, stacked policy/reward column, timeline, and panel shelf reproduce the target hierarchy. The game bitmap measures 659 × 618 CSS pixels in the captured state (ratio 1.06634, matching 256:240 after fractional CSS rounding) and is uniformly fitted rather than stretched.
- Colors and tokens: the near-black/teal surfaces, cyan structure, green live state, magenta event markers, restrained borders, and low elevation match the visual direction without gradients or decorative UI noise.
- Image quality and asset fidelity: the real streamed game and observation canvases remain pixel-sharp, use the incoming bitmap dimensions, and preserve aspect ratio. No generated placeholder art, custom SVG substitute, or CSS illustration replaces application imagery.
- Copy and content: labels are specific to playback and policy interpretation. The removed Environment/Game headings and input hint leave the game region maximally useful. The live/inspection distinction is explicit in the overlay and timeline.
- Icons and affordances: visible text controls (`Drag`, `Panel`, `Fullscreen`) are less minimal than the reference's dot/icon affordances but are clearer without introducing a mismatched icon family. This is acceptable P3 polish, not a usability or fidelity blocker.
- Responsiveness and accessibility: the 1280-pixel desktop grid has no horizontal overflow; the naturally observed 852-pixel viewport stacks panels without collision. Focus-visible styling, semantic buttons, labelled canvases, an accessible range input, keyboard panel movement/resizing, and pointer drag/resize are present.

**Focused Region Evidence**

- The focused comparison checks the dense controls, policy bars, transition statistics, and reward chart at readable scale. The implementation intentionally makes the controls rail narrower to reserve more space for the aspect-correct player; button wrapping remains coherent and policy/reward values remain scannable.

**Primary Interactions Tested**

- Restored Action history from the shelf, then hid it back to the shelf.
- Dragged Policy distribution from column 10/row 1 to column 1/row 16 and restored the default layout.
- Resized Policy distribution from 3 × 7 to 2 × 6 grid units and restored the default layout.
- Scrubbed to a retained transition while the live sequence continued, verified the separate `INSPECTING` state, then returned live.
- Opened a second workspace window, moved Observation and attribution into it, received the latest 336 × 84 observation while paused/live, and observed the same sequence and controller lease in both windows.
- Verified current-frame delivery when a window adds a frame subscription.
- Checked both browser windows for console warnings and errors: none.

**Comparison History**

- Iteration 1 found a P2 interaction gap: native HTML drag did not reliably move a panel under the browser interaction test. It was replaced with pointer-driven dragging plus a live grid drop preview. Post-fix evidence moved Policy distribution to column 1/row 16 and emitted `Policy distribution moved.`
- Iteration 1 found a P2 secondary-window placement gap: restoring the first panel could preserve its old row and leave an empty area above it. Empty windows now place their first panel at column 1/row 1. Post-fix evidence shows Observation and attribution at y=60.07 with grid row `1 / span 8`.
- Iteration 1 found a P2 state-label gap in a playback/recording mode transition. The header eyebrow now resets to `RLAB PLAYER` when playback mode becomes active. The revised browser capture shows the correct label.
- Iteration 2 full-view and focused comparisons found no remaining P0/P1/P2 issues.

**Implementation Checklist**

- [x] Preserve exact incoming game aspect ratio.
- [x] Keep the game, controls, policy, and reward visible together at desktop width.
- [x] Support pointer and keyboard panel movement plus pointer and keyboard resizing.
- [x] Persist named/shared layouts and expose hidden panels through the shelf.
- [x] Support synchronized multi-window telemetry, frame subscriptions, and workspace-level control.
- [x] Keep retained timeline inspection separate from the live runner revision.
- [x] Verify backend, frontend syntax, browser interactions, visual fidelity, and console health.

**Follow-up Polish**

- P3: replace the `Drag` and `Panel` text with a coherent compact icon set only if the product adopts one globally; keep accessible names and tooltips.

final result: passed
