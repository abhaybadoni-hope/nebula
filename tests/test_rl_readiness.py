from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.receiver_search import (
    _sample_actions, preflight_receiver_search, run_receiver_search, summarize_receiver_search,
    verify_receiver_search,
)
from simulator.cache import EvaluationCache
from simulator.channel import (
    ChannelPortMap, S4PChannel, filter_channel, sampling_offset_bits,
    sampling_reference_phase_s,
)
from simulator.config import SimulationConditions, Sky130Config
from simulator.ngspice import NgSpiceConfig
from simulator.provenance import spice_dependency_fingerprint, spice_dependency_manifest
from simulator.receiver import (
    EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, StageResult,
    _transient_violations,
)
from simulator.receiver_metrics import apply_hybrid_one_tap_dfe, choose_sampling_phase, dfe_eye_metrics
from simulator.rl_adapter import (
    ACTION_BOUNDS, CONSTRAINT_NAMES, OBSERVATION_NAMES, RLBudget,
    ReceiverRLAdapter, normalized_action_to_parameters,
)
from simulator.stimulus import NRZStimulusConfig, generate_nrz
from simulator.waveform import Trace, ac_metrics, hd3_db
from experiments.build_compact_sky130 import CORNERS, DEVICE, build_compact_sky130


def _cache_process_write(arguments):
    root, key, worker = arguments
    cache = EvaluationCache(root)
    cache.put(key, {"worker": worker})
    return cache.get(key)["evaluation_id"]


def _evaluation(success=True, *, metrics=None, identity="fake"):
    values = {
        "gain_100mhz_db": -2.0, "gain_2p5ghz_db": 4.0, "peaking_db": 6.0,
        "ctle_power_w": 1e-3, "output_common_mode_v": 0.9,
        "dfe_eye_height_v": 0.2, "dfe_locked_phase_eye_height_v": 0.2,
        "dfe_eye_width_ui": 0.6, "dfe_min_margin_v": 0.1,
        "dfe_error_rate": 0.0, "dfe_error_count": 0,
        "channel_loss_2p5ghz_db": -3.0,
    }
    values.update(metrics or {})
    stage = StageResult("fake", success, 0.0, metrics=values)
    return ReceiverEvaluation(
        success, ReceiverParameters(), SimulationConditions(),
        EvaluationFidelity.TRAINING, (stage,), values,
        None if success else "fake", 0.0, identity, {},
    )


class DelayAndDFEGoldenTests(unittest.TestCase):
    def _channel(self, delay_ui, ui, step, count):
        padded = 1 << (2 * count - 1).bit_length()
        frequency = np.fft.rfftfreq(padded, step)
        transfer = np.exp(-2j * np.pi * frequency * delay_ui * ui)
        matrix = np.zeros((len(frequency), 4, 4), dtype=complex)
        for destination, source, sign in ((2, 0, 1), (2, 1, -1), (3, 0, -1), (3, 1, 1)):
            matrix[:, destination, source] = sign * transfer / 2
        return S4PChannel(Path("analytic.s4p"), frequency, matrix, 50.0, "ri", "analytic")

    def test_pinned_bulk_delays_recover_bits_and_delay(self):
        config = NRZStimulusConfig(bit_count=96, warmup_bits=24, tail_bits=24)
        stimulus = generate_nrz(config)
        self.assertEqual(
            hashlib.sha256(bytes(stimulus.bits.tolist())).hexdigest(),
            "a7b35c2b34fa187f7a8635584e7c099c950597ee6c3155273af85f18f94cb1c4",
        )
        for delay_ui in (0.0, 0.3, 0.7, 0.99, 1.0, 5.0, 20.0):
            with self.subTest(delay_ui=delay_ui):
                channel = self._channel(delay_ui, config.ui_s, config.time_step_s, len(stimulus.time_s))
                self.assertLessEqual(abs(channel.bulk_delay_s() - delay_ui * config.ui_s), config.time_step_s)
                self.assertAlmostEqual(channel.insertion_loss_db(2.345e9), 0.0, places=9)
                self.assertLess(channel.validation_metrics()["channel_negative_time_energy_ratio"], 0.05)
                offset = sampling_offset_bits(channel.bulk_delay_s(), config.ui_s)
                usable = len(stimulus.bits) - offset
                output = filter_channel(channel, stimulus.time_s, stimulus.differential_v)
                selected = choose_sampling_phase(
                    stimulus.time_s, output, stimulus.bits[:usable], config.ui_s,
                    12, 24, config.time_step_s, start_time_s=offset * config.ui_s,
                )
                decisions = (selected.raw_samples_v >= 0).astype(int)
                np.testing.assert_array_equal(decisions[24:usable - 24], stimulus.bits[24:usable - 24])

    def test_integer_offset_is_stable_around_integer_ui(self):
        ui = 200e-12
        for value in (1.0 - 1e-9, 1.0, 1.0 + 1e-9):
            self.assertEqual(sampling_offset_bits(value * ui, ui), 1)

    def test_eye_width_is_invariant_across_fractional_delay_wrap(self):
        config = NRZStimulusConfig(bit_count=64, warmup_bits=16, tail_bits=16)
        stimulus = generate_nrz(config)
        widths = []
        for delay_ui in (0.1, 0.3, 0.5, 0.7, 0.9):
            channel = self._channel(
                delay_ui, config.ui_s, config.time_step_s, len(stimulus.time_s),
            )
            fitted_delay_s = channel.bulk_delay_s()
            offset = sampling_offset_bits(fitted_delay_s, config.ui_s)
            waveform = np.interp(
                stimulus.time_s - delay_ui * config.ui_s,
                stimulus.time_s, stimulus.differential_v,
                left=stimulus.differential_v[0],
            )
            aligned_bits = stimulus.bits[:len(stimulus.bits) - offset]
            reference_phase = sampling_reference_phase_s(
                fitted_delay_s, config.ui_s, offset, config.time_step_s,
            )
            if delay_ui == 0.5:
                self.assertEqual(reference_phase, 0.0)
            sampling = choose_sampling_phase(
                stimulus.time_s, waveform, aligned_bits, config.ui_s,
                8, 16, config.time_step_s,
                start_time_s=offset * config.ui_s,
                reference_phase_s=reference_phase,
            )
            metrics = dfe_eye_metrics(
                stimulus.time_s - offset * config.ui_s, waveform,
                aligned_bits, config.ui_s,
                config.time_step_s, 16, 48 - offset, 0.0,
                locked_phase_s=sampling.phase_s,
            )
            widths.append(metrics["dfe_eye_width_ui"])
        self.assertLessEqual(max(widths) - min(widths), config.time_step_s / config.ui_s + 1e-12)

    def test_phase_training_does_not_use_measurement_bits(self):
        config = NRZStimulusConfig(bit_count=32, warmup_bits=8, tail_bits=8)
        stimulus = generate_nrz(config)
        first = choose_sampling_phase(
            stimulus.time_s, stimulus.differential_v, stimulus.bits, config.ui_s,
            2, 8, config.time_step_s, tap_v=0.05,
        )
        changed_bits = stimulus.bits.copy()
        changed_bits[8:] = 1 - changed_bits[8:]
        second = choose_sampling_phase(
            stimulus.time_s, stimulus.differential_v, changed_bits, config.ui_s,
            2, 8, config.time_step_s, tap_v=0.05,
        )
        self.assertEqual(first.phase_s, second.phase_s)

    def test_dfe_aware_phase_choice_matches_two_phase_oracle(self):
        ui = 200e-12
        bits = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
        signs = np.where(bits > 0, 1.0, -1.0)
        previous = np.r_[-1.0, signs[:-1]]
        phase_a = 0.25 * signs + 0.20 * previous
        phase_b = 0.15 * signs
        time = np.arange(2 * len(bits) + 1, dtype=float) * (ui / 2)
        waveform = np.empty_like(time)
        waveform[0:-1:2] = phase_a
        waveform[1::2] = phase_b
        waveform[-1] = phase_a[-1]
        raw = choose_sampling_phase(time, waveform, bits, ui, 0, len(bits), ui / 2, tap_v=0.0)
        corrected = choose_sampling_phase(time, waveform, bits, ui, 0, len(bits), ui / 2, tap_v=0.20)
        self.assertAlmostEqual(raw.phase_s, ui / 2)
        self.assertAlmostEqual(corrected.phase_s, 0.0)

    def test_controlled_precursor_and_postcursor_isi(self):
        bits = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
        signs = np.where(bits > 0, 1.0, -1.0)
        previous = np.r_[-1.0, signs[:-1]]
        following = np.r_[signs[1:], 1.0]
        samples = 0.30 * signs + 0.08 * previous + 0.04 * following
        result = apply_hybrid_one_tap_dfe(samples, bits, 0.08, training_stop=4)
        self.assertEqual(result.error_count, 0)
        self.assertGreater(float(np.min(result.margins_v)), 0.25)

    def test_port_mapping_reverses_polarity(self):
        config = NRZStimulusConfig(bit_count=8, warmup_bits=2, tail_bits=2)
        stimulus = generate_nrz(config)
        channel = self._channel(0.0, config.ui_s, config.time_step_s, len(stimulus.time_s))
        reversed_channel = S4PChannel(
            channel.path, channel.frequency_hz, channel.matrix, 50.0, "ri", "analytic",
            ChannelPortMap(1, 2, 4, 3).tx_ports, ChannelPortMap(1, 2, 4, 3).rx_ports,
        )
        np.testing.assert_allclose(
            reversed_channel.differential_transfer(), -channel.differential_transfer(),
        )

    def test_acausal_channel_is_detected(self):
        config = NRZStimulusConfig(bit_count=32, warmup_bits=8, tail_bits=8)
        channel = self._channel(0.0, config.ui_s, config.time_step_s, 3201)
        frequency = channel.frequency_hz
        advance = np.exp(2j * np.pi * frequency * 0.4 * config.ui_s)
        matrix = channel.matrix.copy()
        for destination, source, sign in ((2, 0, 1), (2, 1, -1), (3, 0, -1), (3, 1, 1)):
            matrix[:, destination, source] = sign * advance / 2
        acausal = S4PChannel(channel.path, frequency, matrix, 50.0, "ri", "acausal")
        self.assertLess(acausal.bulk_delay_s(), 0.0)


class ShapeHD3AndConstraintTests(unittest.TestCase):
    def _ac(self, kind):
        frequency = np.logspace(6, math.log10(20e9), 1000)
        x = np.log10(frequency)
        if kind == "peak":
            gain = 7.0 * np.exp(-((x - math.log10(2e9)) / 0.45) ** 2)
        elif kind == "shelf":
            gain = 6.0 / (1 + np.exp(-(x - 8.5) * 6))
        else:
            gain = 3.0 * (x - 6.0)
        return ac_metrics(Trace("frequency", frequency, {
            "transfer_mag": 10 ** (gain / 20), "transfer_phase_deg": np.zeros_like(gain),
        }))

    def test_ac_shape_classification(self):
        self.assertEqual(self._ac("peak")["response_class"], "local_peak")
        self.assertEqual(self._ac("shelf")["response_class"], "controlled_shelf")
        self.assertEqual(self._ac("rise")["response_class"], "uncontrolled_rise")

    def test_hd3_known_minus_40dbc_on_jittered_noncoherent_timebase(self):
        rng = np.random.default_rng(7)
        time = np.linspace(100.3e-9, 231.7e-9, 5000)
        time += np.r_[0.0, np.cumsum(rng.uniform(-1e-15, 1e-15, len(time) - 1))]
        time = np.maximum.accumulate(time)
        omega = 2 * np.pi * 100e6 * time
        trace = Trace("time", time, {"vout_diff": (
            np.sin(omega) + 0.2 * np.sin(2 * omega + 0.4) + 0.01 * np.sin(3 * omega + 0.2)
        )})
        self.assertAlmostEqual(hd3_db(trace), -40.0, delta=0.15)
        clean = Trace("time", time, {"vout_diff": np.sin(omega)})
        self.assertLess(hd3_db(clean), -200.0)

    def test_hd3_rejects_large_time_gap(self):
        time = np.linspace(100e-9, 200e-9, 1000)
        time[500:] += 10e-9
        with self.assertRaises(ValueError):
            hd3_db(Trace("time", time, {"vout_diff": np.sin(2 * np.pi * 100e6 * time)}))

    def test_hd3_rejects_short_and_undersampled_records(self):
        short = np.linspace(100e-9, 150e-9, 1000)
        with self.assertRaises(ValueError):
            hd3_db(Trace("time", short, {"vout_diff": np.sin(2 * np.pi * 100e6 * short)}))
        coarse = np.arange(100e-9, 210e-9, 2e-9)
        with self.assertRaises(ValueError):
            hd3_db(Trace("time", coarse, {"vout_diff": np.sin(2 * np.pi * 100e6 * coarse)}))

    def _physical_metrics(self, **updates):
        values = {
            "dfe_error_count": 0, "dfe_min_margin_v": 1e-12,
            "transient_output_min_v": -0.01, "transient_output_max_v": 1.81,
            "transient_supply_v": 1.8, "transient_average_power_w": 14.999e-3,
            "dfe_locked_phase_eye_height_v": 0.11, "dfe_eye_width_ui": 0.41,
        }
        values.update(updates)
        return values

    def test_exact_ber_margin_rail_and_power_boundaries(self):
        self.assertEqual(_transient_violations(self._physical_metrics(), EvaluationFidelity.CANDIDATE), [])
        cases = (
            {"dfe_error_count": 1}, {"dfe_min_margin_v": 0.0},
            {"transient_output_min_v": -0.010001}, {"transient_output_max_v": 1.810001},
            {"transient_average_power_w": 0.015},
        )
        for update in cases:
            self.assertTrue(_transient_violations(self._physical_metrics(**update), EvaluationFidelity.TRAINING))

    def test_inactive_resistive_output_may_return_close_to_vdd(self):
        metrics = self._physical_metrics(
            transient_output_min_v=1.20,
            transient_output_max_v=1.799,
        )
        self.assertEqual(_transient_violations(metrics, EvaluationFidelity.TRAINING), [])


class CacheAndRLContractTests(unittest.TestCase):
    def test_constraint_aware_search_sampling_is_deterministic_and_feasible(self):
        conditions = SimulationConditions()
        first = _sample_actions(20, 19, "constraint_aware_v1", conditions)
        self.assertEqual(first, _sample_actions(20, 19, "constraint_aware_v1", conditions))
        self.assertEqual(len(set(first)), 20)
        for action in first:
            parameters = normalized_action_to_parameters(action)
            common_mode = conditions.supply_v - 0.5 * parameters.rload_ohm * parameters.itail_a
            self.assertGreaterEqual(common_mode, 0.25)
            self.assertLessEqual(common_mode, conditions.supply_v - 0.25)
            self.assertGreaterEqual(parameters.rdeg_ohm * parameters.cdeg_f, 0.1e-9)
            self.assertLessEqual(parameters.rdeg_ohm * parameters.cdeg_f, 1.0e-9)
            self.assertLessEqual(abs(parameters.dfe_tap_v), 0.2)

    def test_compact_sky130_builder_pins_official_device_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sky130A"
            ngspice = root / "libs.tech" / "ngspice"
            device_root = root / "libs.ref" / "sky130_fd_pr" / "spice"
            ngspice.mkdir(parents=True)
            device_root.mkdir(parents=True)
            full = ngspice / "sky130.lib.spice"
            full.write_text("* full model selector\n", encoding="utf-8")
            mismatch = device_root / f"{DEVICE}__mismatch.corner.spice"
            mismatch.write_text(".param mismatch=0\n", encoding="utf-8")
            for corner in CORNERS:
                (device_root / f"{DEVICE}__{corner}.pm3.spice").write_text(
                    f"* {corner} model\n", encoding="utf-8",
                )

            compact = build_compact_sky130(full)
            text = compact.read_text(encoding="utf-8")
            for corner in CORNERS:
                self.assertIn(f".lib {corner}", text)
                self.assertIn(f"{DEVICE}__{corner}.pm3.spice", text)
                self.assertIn(f".endl {corner}", text)
            self.assertEqual(text.count(".option scale=1.0u"), len(CORNERS))
            self.assertEqual(text.count(f"{DEVICE}__mismatch.corner.spice"), len(CORNERS))

    def test_model_fingerprint_covers_transitive_spice_includes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.lib.spice"
            corner = root / "corner.spice"
            leaf = root / "leaf.spice"
            model.write_text('.include "corner.spice"\n', encoding="utf-8")
            corner.write_text('.include "leaf.spice"\n', encoding="utf-8")
            leaf.write_text('.model device nmos level=1\n', encoding="utf-8")
            first = spice_dependency_fingerprint(model)
            self.assertEqual(len(spice_dependency_manifest(model)), 3)
            leaf.write_text('.model device nmos level=1 vto=0.7\n', encoding="utf-8")
            self.assertNotEqual(first, spice_dependency_fingerprint(model))

    def test_receiver_search_preflight_fingerprints_external_inputs(self):
        channel_path = Path(__file__).resolve().parents[1] / "channels" / "synthetic_regression.s4p"
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "sky130.lib.spice"
            model.write_text("* preflight fixture", encoding="utf-8")
            report = preflight_receiver_search(
                channel_path=channel_path,
                sky130=Sky130Config(model),
                ngspice=NgSpiceConfig(executable=Path(__file__)),
            )
        self.assertTrue(report["ready_for_search"])
        self.assertEqual(report["channel_port_map"]["tx_positive"], 1)
        self.assertEqual(len(report["channel_checksum"]), 64)
        self.assertEqual(len(report["sky130_model_checksum"]), 64)
        self.assertLessEqual(report["channel_metrics"]["channel_max_singular_value"], 1.01)

    def test_concurrent_cache_writes_same_and_mixed_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = [(directory, "same", index) for index in range(24)]
            arguments += [(directory, f"mixed{index % 4}", index) for index in range(24)]
            with ThreadPoolExecutor(max_workers=12) as pool:
                thread_ids = list(pool.map(_cache_process_write, arguments))
            with ProcessPoolExecutor(max_workers=4) as pool:
                process_ids = list(pool.map(_cache_process_write, arguments[:16]))
            self.assertTrue(all(thread_ids))
            self.assertEqual(set(process_ids), {"same"})
            leftovers = list(Path(directory).rglob("*.tmp")) + list(Path(directory).rglob("*.lock"))
            self.assertEqual(leftovers, [])

    def test_rl_action_endpoints_shapes_budget_and_seed(self):
        low = normalized_action_to_parameters([-1] * len(ACTION_BOUNDS))
        high = normalized_action_to_parameters([1] * len(ACTION_BOUNDS))
        self.assertAlmostEqual(low.rload_ohm, 100.0)
        self.assertAlmostEqual(high.rload_ohm, 10_000.0)

        def evaluator(*args, **kwargs):
            return _evaluation(identity="rl")

        first = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(1), seed=23)
        second = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(1), seed=23)
        self.assertEqual(first.sample_action(), second.sample_action())
        step = first.step([0] * len(ACTION_BOUNDS))
        self.assertEqual(len(step.observation), len(OBSERVATION_NAMES))
        self.assertEqual(len(step.constraints), len(CONSTRAINT_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in (*step.observation, *step.constraints, step.reward)))
        self.assertTrue(step.truncated)
        first.reset(seed=23)
        with self.assertRaises(RuntimeError):
            first.step([0] * len(ACTION_BOUNDS))
        with self.assertRaises(ValueError):
            ReceiverRLAdapter(evaluator=evaluator, fidelity=EvaluationFidelity.SCREENING)

    def test_rl_failed_design_continues_and_missing_metrics_are_marked_invalid(self):
        def evaluator(*args, **kwargs):
            missing = {key: float("nan") for key in OBSERVATION_NAMES}
            missing["dfe_locked_phase_eye_height_v"] = float("nan")
            return _evaluation(False, metrics=missing)

        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(2))
        step = adapter.step([0] * len(ACTION_BOUNDS))
        self.assertFalse(step.terminated)
        half = len(step.observation) // 2
        self.assertTrue(all(value == 0.0 for value in step.observation[half:]))
        self.assertFalse(step.truncated)
        second = adapter.step([0] * len(ACTION_BOUNDS))
        self.assertTrue(second.truncated)

    def test_receiver_search_resumes_from_manifest(self):
        calls = []

        def evaluator(parameters, *args, **kwargs):
            calls.append(parameters)
            return _evaluation(identity=f"id{len(calls)}")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search.jsonl"
            first = run_receiver_search(count=3, seed=5, output=output, evaluator=evaluator)
            second = run_receiver_search(count=3, seed=5, output=output, evaluator=evaluator)
            self.assertEqual(len(first), 3)
            self.assertEqual(second, [])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)
            self.assertTrue(output.with_suffix(".jsonl.manifest.json").is_file())
            summary = summarize_receiver_search(output)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_candidates"], 3)
            self.assertEqual(summary["successful_candidates"], 3)
            self.assertEqual(summary["unique_evaluation_ids"], 3)
            self.assertEqual(summary["missing_candidate_indices"], [])
            self.assertEqual(summary["top_candidates"][0]["candidate_index"], 0)

            def replay_evaluator(parameters, *args, **kwargs):
                return _evaluation(identity=f"id{calls.index(parameters) + 1}")

            verification = verify_receiver_search(
                output, top=2, evaluator=replay_evaluator,
            )
            self.assertTrue(verification["passed"])
            self.assertEqual(verification["verified_candidates"], 2)

            row = output.read_text(encoding="utf-8").splitlines()[0]
            altered = json.loads(row)
            altered["manifest_id"] = "wrong"
            output.write_text(json.dumps(altered) + "\n" + "\n".join(
                output.read_text(encoding="utf-8").splitlines()[1:]
            ), encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize_receiver_search(output)


if __name__ == "__main__":
    unittest.main()
