# Mentor-led corner study

The current study interprets the mentor note as follows:

- Start at TT, 1.8 V and 27 C, with 5 Gb/s NRZ and 2.5 GHz Nyquist frequency.
- Use 6 dB as a provisional preferred boost, with the competition 3-12 dB range still checked.
- Prioritize valid HD3 and eye measurements, then compare distortion, eye height and width. Peaking proximity is a tie breaker.
- Report power and estimated area without using them to rank candidates. Do not remove physical operating-point or waveform validity checks.
- Test the chosen TT parameters unchanged at SS, then FF. Independent per-corner retuning has not been authorized or assumed.
- NR means Nyquist frequency: 2.5 GHz for 5 Gb/s NRZ. Report gain at 2.5 GHz and boost relative to 100 MHz separately from the maximum boost in the 1.25-2.5 GHz band.

This separate diagnostic study does not weaken production acceptance, change PPO rewards, or claim full competition compliance. It uses the legacy CTLE with fixed W=10 um, L=0.15 um, ideal tail current, and behavioral one-tap DFE. The physical sampler remains experimental.

```powershell
python -m experiments.mentor_corner_study --design results/design_a_final_specification.json --output generated/mentor-study-new --seconds 180
```

The study compares the supplied design with 1.5x degeneration resistance and 0.7x degeneration capacitance at TT. It checks the best-ranked candidate at SS and FF with the same settings. Two workers share a wall-clock deadline, with 30-second per-stage limits. This is a small local comparison, not global optimization.

Outputs include all stage failures/provenance in study.json, retained measured waveforms in NPZ files, actual CTLE eye overlays in SVG, and report.html. The plot is explicitly before DFE; tabulated eye metrics use behavioral DFE. Eye checks use 128-bit PRBS7 over the repository synthetic channel. HD3 uses 100 MHz and 100 mV peak-to-peak differential input, matching the existing bench; the competition slide does not explicitly settle peak versus peak-to-peak convention. Noise is not measured in this study.

An initial 512-bit comparison exceeded 30-second transient timeouts. Those missing measurements must not be interpreted as eye failures or passes. Longer-pattern, full PVT, measured-channel and input-amplitude-convention validation remains necessary.
