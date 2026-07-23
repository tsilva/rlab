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
