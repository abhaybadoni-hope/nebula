# Noise and tuning evidence

The selected mentor-study design passes input-referred noise at TT/SS/FF, 1.8 V and 27 C: approximately 0.364/0.386/0.340 mVrms, integrated over 10 MHz-5 GHz. This is CTLE noise from the ideal-tail schematic, not total physical receiver noise.

A bounded 36-point TT load-resistance/degeneration-capacitance sweep found local peaks near targets 1.25, 1.875 and 2.5 GHz using a declared 5% frequency tolerance. The measured peaks are 1.288, 1.862 and 2.399 GHz. Each selected setting passed DC/AC, noise, HD3 at 100 MHz/100 mVpp differential, and 128-bit PRBS7 eye checks on the synthetic channel. All other CTLE parameters remained fixed.

These are sampled tuning settings. They do not establish exact endpoint coverage, continuous tunability, or tuning performance across PVT. The base-design corner noise test and the TT tuning tests are separate evidence.

```powershell
python -m experiments.noise_tuning_study --design generated/mentor-study-128/selected.json --output generated/noise-tuning-new
```

The study uses two workers for DC/AC screening, then validates only selected settings. Each simulator invocation is capped at 30 seconds. This offline study is not part of RL training or ordinary inference.

A noise-bench parameter collision was repaired: RCM_BIAS now names the input common-mode bias resistors, while RBIAS remains the physical-tail mirror bias setting. The legacy base-design noise numbers were remeasured after this change. Tail mirror sizing can no longer silently override the input bias resistor value or vice versa.

Machine-readable results are in noise-tuning-results.json; complete per-stage measurements and provenance are in generated/noise-tuning-study/study.json. The HTML report is in the same generated directory. Exact endpoints, long-pattern tests for every tuning setting and full PVT remain outstanding.
