# Finalist HD3 and long-eye validation

The selected mentor-study CTLE is tested separately from training. The existing 100 mVpp differential, 100 MHz HD3 bench is compared with 200 mVpp (100 mV peak differential), so either interpretation of the slide's amplitude can be inspected. The official convention still needs confirmation.

For TT, SS and FF at 1.8 V and 27 C, both amplitudes are checked with the existing 20 ps maximum timestep, tighter tolerances at 10 ps, and tighter tolerances at 5 ps. Tight settings are reltol=1e-5, abstol=1e-12 and vntol=1e-8. Each waveform is independently analyzed with steady-state discard windows of 100, 200 and 250 ns. All runs end at 500 ns. Stability within 0.1 dB is an engineering convergence criterion, not an official competition requirement.

The 18 simulations and 54 window measurements remained below -30 dB. The largest spread within an amplitude/corner combination was below 0.003 dB. At 100 mVpp the worst result was about -82.33 dB; at 200 mVpp it was about -70.22 dB. These results validate numerical stability within this circuit model; they do not validate a physical sampler or extracted layout.

```powershell
python -m experiments.verify_hd3_eye --design generated/mentor-study-128/selected.json --output generated/verification-new --eye-timeout 300
```

The eye checks are serial and offline, using the original 2 ps timestep and the synthetic channel. --eye-timeout applies to each simulator process, not to the entire validation campaign. The 512-bit uncompressed TT check passed with about 1.411 V DFE eye height, 0.85 UI width and zero measured bit errors in 69 seconds. A 1024-bit run exceeded 120 seconds; timeout means missing evidence, not circuit failure.

Optional --pwl-tolerance bounds the interpolation error of a reduced differential stimulus waveform before source-division compensation. Default zero preserves every original point. A 1 microvolt experiment removed only about 4% of points and did not establish a useful speedup; it is not enabled in training or ordinary validation. --skip-hd3 is only for eye-only reruns and reports no HD3 verification for that run.

Remaining limits: legacy fixed-dimension CTLE with ideal tail current, behavioral DFE, synthetic channel, nominal voltage and temperature. Noise, SF/FS, supply/temperature variation, long PRBS15 and physical implementation still require separate validation.

Long validation length overrides now apply only to candidate/final receiver waveforms. The early CTLE diagnostic stays at 32 bits and ordinary training stays at its configured 128 bits; previously a long validation override unnecessarily expanded the early diagnostic too.

The separate validate_finalist command now exposes --stage-timeout (default 300 seconds), still capped by its remaining total budget. This allows long final transients without changing the 30-second quick-search stage timeout.

The subsequent uncompressed 1024-bit retry also exceeded 300 seconds. It remains unverified; no 1024-bit eye pass is claimed. The completed 512-bit waveform is retained and plotted in generated/finalist-verification/report.html. Compression runs also timed out and were not enabled by default.
