# Collapsed playback toggle design QA

## Evidence

- Source visual truth: `/var/folders/wz/x29jb7_x5rdc_5dcjr4qnhg00000gn/T/codex-clipboard-60cacafa-dc4c-404e-8065-8c394c57c32d.png`
- Browser-rendered implementation: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/player-preview-full.png`
- Browser-rendered running state: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/player-preview-running-full.png`
- Full component comparison: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/playback-toggle-comparison.png`
- Source pixels: 164 × 120 at 2× density, normalized to 82 × 60 CSS pixels.
- Implementation screenshot pixels: 1265 × 712. Focused comparison crops: 82 × 60 CSS pixels at browser device pixel ratio 2.
- State: dark desktop player, paused and playing states, human driver.

## Findings

- No actionable P0, P1, or P2 differences remain.
- The requested structural difference is present: the reference's separate pause and play controls are replaced by one control in the same first transport slot.
- In the paused state, the single control uses the reference's filled teal play treatment and play icon.
- In playing, stepping, and continuing states, the same control switches to the reference's dark secondary treatment and pause icon.
- Fonts and typography: the uppercase Playback label, weight, tracking, size, and line height remain unchanged.
- Spacing and layout rhythm: the button retains the existing 31 × 34 CSS-pixel footprint; the transport grid now uses five equal columns so no empty sixth slot remains.
- Colors and visual tokens: the play and pause states reuse the existing primary and neutral button tokens from the reference.
- Image quality and asset fidelity: both states reuse the packaged Tabler play and pause symbols; no substitute glyphs, CSS drawings, or new raster assets were introduced.
- Copy and content: the visible label is unchanged. The accessible name and tooltip now follow the action currently available.

## Comparison History

- Pass 1 compared the normalized source with both browser-rendered toggle states in one composite. The only structural difference is the explicitly requested collapse from two buttons to one state-aware button. No P0/P1/P2 mismatches were found, so no visual correction loop was needed.

## Primary Interactions Tested

- Clicked Play and verified that the one control changed to command `pause`, accessible name `Pause`, the pause tooltip, the pause icon, and the neutral treatment.
- Clicked Pause and verified that the same control changed back to command `play`, accessible name `Play`, the play tooltip, the play icon, and the primary treatment.
- Confirmed there is exactly one visible Play or Pause control in each state.
- Checked the browser console after both transitions: no errors.

## Implementation Checklist

- [x] Replace separate Play and Pause buttons with one state-aware control.
- [x] Preserve the existing icon pack, visual treatments, tooltip behavior, and accessible naming.
- [x] Treat playing, stepping, and continuing as active playback states.
- [x] Reflow the transport controls into five equal columns.
- [x] Verify both directions in the actual browser-rendered player.

final result: passed

---

# Timeline label stacking design QA

**Comparison target**

- Source visual truth: `/var/folders/wz/x29jb7_x5rdc_5dcjr4qnhg00000gn/T/codex-clipboard-80b68e66-677b-4119-afab-f6250e25ab83.png`
- Browser-rendered implementation: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/timeline-label-above-live.png`
- Normalized implementation crop: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/timeline-label-above.png`
- Focused side-by-side comparison: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/timeline-label-comparison.png`
- Browser viewport: 1292 × 900 CSS px at 1× density.
- Source pixels: 1292 × 454. Implementation pixels: 1292 × 900; the timeline region was cropped to 1292 × 454 with no scaling for the focused comparison.
- State: dark desktop player, loaded Breakout session at episode 3, step 0, sequence 1072.

**Full-view comparison evidence**

- The browser-rendered player preserves the existing dashboard proportions, timeline border treatment, cyan status text, neutral scrubber track, and page spacing.
- The timeline is now a single-column grid. Its status row and track both measure 1236.97 CSS px within the same container, so status-text length no longer participates in sizing the scrubber.
- The document has no horizontal overflow.

**Focused comparison evidence**

- The side-by-side comparison shows the source label sharing a row with the track and the implementation label occupying a separate row directly above the full-width track.
- The implementation keeps a compact 3.52 CSS-pixel vertical gap, preserving the original control density while removing the width coupling.

**Required fidelity surfaces**

- Fonts and typography: the existing cyan monospace family, weight, size, line height, and uppercase status copy remain unchanged; overflow now truncates with an ellipsis instead of resizing the track.
- Spacing and layout rhythm: the label and scrubber are stacked in one column, aligned to the same left and right bounds, with a compact vertical gap.
- Colors and visual tokens: the existing timeline borders, cyan accent, neutral track, and dark surface tokens are unchanged.
- Image quality and asset fidelity: no image or icon assets are involved in this layout-only change.
- Copy and content: status fields and scrubber accessible labeling are unchanged.

**Findings**

- No remaining actionable P0, P1, or P2 mismatch for the requested label-above-track layout.

**Comparison history**

- Initial P2: the label and scrubber shared a two-column grid, so changing the label width changed the scrubber width.
- Fix: converted the timeline to one `minmax(0, 1fr)` column, stacked the label above the track, and constrained label overflow independently.
- Post-fix evidence: the focused comparison and browser measurements show equal label/track column widths and a full-width scrubber independent of label content.

**Interaction and runtime verification**

- Verified the scrubber retains a one-step increment and its full available track width through the packaged player test.
- Browser console errors: none.
- Automated player tests: 13 passed.

**Implementation checklist**

- [x] Place the timeline status label above the scrubber.
- [x] Make scrubber width independent from status-label width.
- [x] Preserve one-step scrubbing and existing visual tokens.
- [x] Verify the rendered layout, horizontal overflow, console, and focused tests.

**Follow-up polish**

- None required for this change.

final result: passed

---

# Dropdown chevron spacing design QA

**Comparison target**

- Source visual truth: `/var/folders/wz/x29jb7_x5rdc_5dcjr4qnhg00000gn/T/codex-clipboard-62cbebaa-13f5-49ae-8190-0bbc4c43ca2a.png`
- Browser-rendered implementation: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/rlab-select-chevron-after-matched.png`
- Focused side-by-side comparison: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/rlab-select-chevron-comparison.png`
- Viewport: 1265 × 712 CSS px at 1× density.
- Source pixels: 586 × 286. Implementation pixels: 1265 × 712; the Live Signals region was normalized to 586 × 286 for the focused comparison.
- State: dark theme, paired Stats workspace, empty Live Signals history, signal dropdown showing “Choose a signal.”

**Full-view comparison evidence**

- The implementation retains the existing panel grid, typography, borders, colors, control height, and empty-state content. The only visible control change is the dropdown chevron placement and its reserved text space.

**Focused comparison evidence**

- The matched-width select measures 509 CSS px versus approximately 512 px in the source.
- The chevron is positioned 13.6 CSS px from the right edge, with 35.2 CSS px reserved for the icon and clearance. It no longer crowds the border or overlaps long option text.

**Required fidelity surfaces**

- Fonts and typography: unchanged from the existing player; label and option hierarchy remain consistent.
- Spacing and layout rhythm: corrected right-edge chevron inset; field dimensions and surrounding grid spacing remain unchanged.
- Colors and visual tokens: existing dark surface, border, and foreground tokens are preserved.
- Image quality and asset fidelity: the chevron uses a crisp Tabler icon asset matching the player’s existing icon family.
- Copy and content: unchanged.

**Findings**

- No remaining actionable P0, P1, or P2 mismatch for the requested dropdown-arrow spacing.

**Comparison history**

- Initial P2: the native select arrow sat too close to the right border.
- Fix: replaced the browser-native arrow with the project’s Tabler chevron, inset it by 0.85rem, and reserved 2.2rem of right padding.
- Post-fix evidence: the focused side-by-side comparison shows clear right-edge breathing room at the matched field width.

**Interaction and runtime verification**

- Playback Sampling successfully changed from stochastic to deterministic after the custom select styling was applied.
- Browser console errors: none.
- Automated player tests: 13 passed.

**Implementation checklist**

- [x] Apply shared select styling.
- [x] Preserve dropdown interaction.
- [x] Verify Live Signals at the reference field width.
- [x] Check the browser console and player test suite.

**Follow-up polish**

- None required for this change.

final result: passed

---

# Right-aligned header status design QA

**Comparison target**

- Source visual truth: `/var/folders/wz/x29jb7_x5rdc_5dcjr4qnhg00000gn/T/codex-clipboard-06dbe56a-1cd0-4054-ae99-008f7f8826a6.png`
- Browser-rendered implementation: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/header-status-right-live.png`
- Normalized source header: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/header-status-source-normalized.png`
- Implementation header crop: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/header-status-right-crop.png`
- Focused stacked comparison: `/Users/tsilva/.codex/visualizations/2026/07/23/019f8f0d-dd60-79a3-9345-8650130dbcfd/header-status-comparison.png`
- Browser viewport: 1372 × 700 CSS px at 1× density.
- Source pixels: 2744 × 236 at 2× density. Its 2744 × 96 app-header region was normalized to 1372 × 48 CSS pixels.
- Implementation pixels: 1372 × 700 at 1× density. Its app-header region is 1372 × 48 pixels.
- State: dark desktop player with a live synced connection. The source is the Controller window; the implementation capture is an Observer window, which changes only the final status badge copy.

**Full-view comparison evidence**

- The implementation preserves the product identity at the left and keeps the existing window controls, sampling badge, and controller/observer badge at the right.
- The header is now a two-column grid: a flexible brand column and one intrinsic right-side controls/status column.
- At both 1372 and 720 CSS pixels, the document has no horizontal overflow and Synced remains in the right-side cluster.

**Focused comparison evidence**

- The normalized comparison shows Synced centered in the source header and immediately after the window controls in the implementation header.
- Synced, Sampling, and Controller/Observer now read as one right-aligned status cluster without changing their individual typography, color, borders, or live state.

**Required fidelity surfaces**

- Fonts and typography: the existing status font sizes, weights, and badge hierarchy are unchanged.
- Spacing and layout rhythm: Synced now uses the same 0.58rem flex gap as the other right-side header items; the unused center grid column was removed.
- Colors and visual tokens: the green synced state and existing muted/active badge tokens are unchanged.
- Image quality and asset fidelity: existing packaged header icons are unchanged; no new image assets were needed.
- Copy and content: connection, sampling, and controller/observer labels retain their existing live values.

**Findings**

- No remaining actionable P0, P1, or P2 mismatch for the requested right-aligned status grouping.

**Comparison history**

- Initial P2: Synced occupied a dedicated centered header column, separating it from the Sampling and Controller statuses.
- Fix: moved the existing live connection element into the right-side status container and reduced the header to two grid columns.
- Post-fix evidence: the focused comparison shows all three statuses in the right cluster, while browser measurements show no overflow at wide or narrow widths.

**Interaction and runtime verification**

- Verified a live connection updates the relocated element to Synced.
- Verified responsive visibility and zero horizontal overflow at 1372 and 720 CSS pixels.
- Browser console errors: none.
- Automated player tests: 13 passed.

**Implementation checklist**

- [x] Move Synced into the right-side header status cluster.
- [x] Preserve live connection-state updates and accessibility announcements.
- [x] Remove the unused center header column and obsolete responsive rules.
- [x] Verify wide and narrow layouts, browser console, and focused tests.

**Follow-up polish**

- None required for this change.

final result: passed
