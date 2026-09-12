# Strict validation

Candidate acceptance is separate from the PPO terminal reward. Eye height,
eye width, positive decision margin, and power must strictly satisfy both
the competition limits and the requested target. Peaking accepts the inclusive
3–12 dB band. Missing and non-finite measurements cannot establish acceptance.
The PPO reward and training termination mechanics are unchanged.

When final validation is requested with `--measure-hd3-noise`, every selected
candidate is checked at FINAL fidelity. A simulator failure, missing final
measurement, or constraint violation rejects it. The pipeline tries the next
candidate, retains validation attempts and their measured failures, and exports
only a passing candidate. If none passes, selection is null and no new schematic
or final specification is produced. Previously created files are not deleted.
PVT results are reused within the run when a candidate is rejected.

PVT reports retain actual condition identities. Nominal, smoke, and 27-point
results cannot receive a full-grid PASS. The 27-point option omits SF and FS.
`--pvt-condition-set full60` covers TT/SS/FF/SF/FS, three supply voltages, and
0/27/75/125 C. Any failed tested condition is reported as FAIL; incomplete
passing coverage is NOT CLAIMED with missing coverage explained. All candidates
failing the requested PVT acceptance threshold produce no selected design.

These checks establish only the measured Stage 1 constraints. Frequency
tunability, physical DFE power, and total area are not established by this patch.
Final validation remains opt-in; nominal-only reports retain NOT CLAIMED for
unmeasured noise, HD3, area, and PVT.

Run focused tests with:

```console
python -m unittest tests.test_strict_acceptance tests.test_final_specification tests.test_run_autockt_pipeline tests.test_web_ui
```

The synthetic dry-run backend checks only its four modeled target metrics;
it does not establish peaking, HD3, noise or circuit compliance. Unmodeled
metrics are omitted from its candidate reports. Real candidates require peaking.
