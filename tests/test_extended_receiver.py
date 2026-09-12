import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from simulator.sizing import CircuitSizing
from simulator.config import SimulationConditions
from simulator.receiver import ReceiverParameters, ReceiverEvaluation, EvaluationFidelity, circuit_block
from simulator.runtime import SearchLimits, BudgetedEvaluator, SearchBudgetExpired
from simulator.stimulus import NRZStimulusConfig, generate_nrz, generate_bits
from simulator.waveform import Trace, hd3_db, integrated_input_noise
from simulator.physical_receiver import render_bench
from analysis.physical_area import estimate_area
from experiments.export_bundle import flatten_model, export_bundle
from experiments.validate_finalist import scenarios
from rl.sized_receiver import decode, BOUNDS, SizedReceiverEnv, STATE_DIM, save_policy, load_policy
from rl.target_spec import TargetSpec
from rl.ppo_agent import PPOAgent

class ExtendedReceiverTests(unittest.TestCase):
    def test_tuning_requires_real_local_peak_and_success(self):
        from experiments.noise_tuning_study import tuning_match
        row={"dc":{"success":True},"ac":{"success":True,"metrics":{"is_local_peak":1,"peak_frequency_hz":2.45e9}}}
        self.assertTrue(tuning_match(row,2.5e9))
        row["ac"]["metrics"]["is_local_peak"]=0
        self.assertFalse(tuning_match(row,2.5e9))
        row["ac"]["metrics"].update(is_local_peak=1,peak_frequency_hz=float("nan"))
        self.assertFalse(tuning_match(row,2.5e9))

    def test_noise_bench_separates_input_bias_from_tail_bias(self):
        from simulator.receiver import BENCHES
        text=(BENCHES/"ctle_noise.cir").read_text()
        self.assertIn("RBIASP inp cm {RCM_BIAS}",text)
        self.assertIn("RBIAS={RBIAS}",text)
        self.assertNotIn("RBIAS=1g",text)

    def test_long_validation_does_not_expand_early_screening(self):
        from simulator.receiver import _transient_config
        c=SimulationConditions(validation_bit_count=4096)
        self.assertEqual(_transient_config(EvaluationFidelity.SCREENING,c).bit_count,32)
        self.assertEqual(_transient_config(EvaluationFidelity.TRAINING,c).bit_count,128)
        self.assertEqual(_transient_config(EvaluationFidelity.FINAL,c).bit_count,4096)

    def test_pwl_compression_bounds_error_and_preserves_edges(self):
        from simulator.stimulus import compress_pwl
        time=np.linspace(0,1,10001)
        signal=np.sin(time*50)*np.exp(-time)+np.where(time>.4,.2,0)
        t,v=compress_pwl(time,signal,1e-6)
        self.assertLess(len(t),len(time))
        self.assertLessEqual(np.max(np.abs(np.interp(time,t,v)-signal)),1e-6+1e-14)
        self.assertEqual((t[0],t[-1]),(time[0],time[-1]))
        t,v=compress_pwl(time,signal,0)
        np.testing.assert_array_equal(v,signal)
        for invalid in (-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError): compress_pwl(time,signal,invalid)

    def test_hd3_convergence_requires_complete_finite_measurements(self):
        from experiments.verify_hd3_eye import summarize_hd3
        rows=[{"corner":"tt","differential_pp_v":.1,"maximum_timestep_ps":step,
               "tight_tolerances":tight,"simulation_success":True,
               "hd3_by_discard_ns":{"100":-40.,"200":-40.,"250":-40.}}
              for step,tight in ((20,False),(10,True),(5,True))]
        self.assertTrue(summarize_hd3(rows)[0]["passes_hd3"])
        self.assertFalse(summarize_hd3(rows[:-1])[0]["passes_hd3"])
        rows[0]["hd3_by_discard_ns"]["100"]=float("nan")
        self.assertFalse(summarize_hd3(rows)[0]["passes_hd3"])

    def test_nyquist_boost_is_not_peak_boost(self):
        from simulator.waveform import Trace, ac_metrics
        f=np.unique(np.r_[np.geomspace(1e6,1e10,501),1e8,2e9,2.5e9])
        gain=3+8*np.exp(-((np.log10(f)-np.log10(2e9))/.15)**2)
        metrics=ac_metrics(Trace("frequency",f,{"transfer_mag":10**(gain/20),"transfer_phase_deg":np.zeros_like(f)}))
        self.assertEqual(metrics["nyquist_frequency_hz"],2.5e9)
        self.assertAlmostEqual(metrics["nyquist_boost_db"],metrics["gain_2p5ghz_db"]-metrics["gain_100mhz_db"])
        self.assertGreater(metrics["peaking_db"],metrics["nyquist_boost_db"]+1)

    def test_mentor_priority_does_not_reward_missing_hd3_or_low_power(self):
        from experiments.mentor_corner_study import priority
        metrics={"hd3_db":-40.,"dfe_locked_phase_eye_height_v":.2,"dfe_eye_width_ui":.6,
                 "dfe_min_margin_v":.1,"dfe_error_count":0,"peaking_db":6.,"ctle_power_w":.02}
        good={"metrics":metrics,"stages":[]}
        missing={"metrics":{k:v for k,v in metrics.items() if k!="hd3_db"},"stages":[]}
        self.assertLess(priority(good),priority(missing))
        self.assertEqual(priority(good),priority({"metrics":{**metrics,"ctle_power_w":.001},"stages":[]}))

    def test_parallel_rollouts_preserve_episode_boundaries(self):
        from rl.parallel_rollout import collect_parallel
        from rl.autockt_env import AutoCktStep
        from threading import Barrier, get_ident
        barrier=Barrier(2)
        caller=get_ident()
        class Agent:
            def act(self,state):
                self.assert_caller()
                return (1,),0.,7.
            def assert_caller(self):
                if get_ident()!=caller: raise AssertionError("policy accessed in worker")
        class Env:
            def __init__(self,index): self.index=index; self.steps=0
            def reset(self): return (self.index,0),{"target":{}}
            def step(self,choices):
                barrier.wait(timeout=5)
                self.steps+=1
                return AutoCktStep((self.index,self.steps),1.,False,self.steps==2,{})
        rows=collect_parallel(Env,Agent(),episodes=2,workers=2)
        self.assertEqual([r.state[0] for r in rows],[0,0,1,1])
        self.assertEqual([r.truncated for r in rows],[False,True,False,True])
        self.assertEqual([r.bootstrap_value for r in rows],[0.,7.,0.,7.])

    def test_parallel_evaluator_never_exceeds_budget(self):
        from concurrent.futures import ThreadPoolExecutor
        def fake(p,c,f,**kwargs):
            return ReceiverEvaluation(False,p,c,f,(),{},"dc",0.,"id",{})
        with TemporaryDirectory() as temp:
            evaluator=BudgetedEvaluator(SearchLimits(10,1),cache_root=temp,evaluator=fake)
            def call():
                try:
                    evaluator(ReceiverParameters(),SimulationConditions(),EvaluationFidelity.TRAINING)
                    return True
                except SearchBudgetExpired: return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes=list(pool.map(lambda _:call(),range(2)))
            self.assertEqual(sum(outcomes),1)
            self.assertEqual(evaluator.calls,1)

    def test_short_training_waveform_does_not_shorten_final_validation(self):
        from simulator.receiver import _transient_config
        conditions=SimulationConditions(training_bit_count=96)
        conditions.validate()
        training=_transient_config(EvaluationFidelity.TRAINING,conditions)
        self.assertEqual((training.bit_count,training.warmup_bits,training.tail_bits),(96,16,16))
        self.assertEqual(_transient_config(EvaluationFidelity.FINAL,conditions).bit_count,1024)
        self.assertEqual(_transient_config(EvaluationFidelity.SCREENING,conditions).bit_count,32)
        self.assertEqual(_transient_config(EvaluationFidelity.TRAINING,SimulationConditions()).bit_count,128)
        for value in (32,129,96.5,True):
            with self.assertRaises(ValueError): SimulationConditions(training_bit_count=value).validate()

    def test_larger_steps_and_clipping(self):
        def fake(p,c,f):
            return ReceiverEvaluation(False,p,c,f,(),{},"dc",0.,"id",{})
        env=SizedReceiverEnv((TargetSpec.from_existing_thresholds(),),evaluator=fake)
        env.reset()
        env.action=[0.] * len(BOUNDS)
        env.step([2,0,1]+[2]*(len(BOUNDS)-3))
        self.assertEqual(env.action[:3],[.2,-.2,0.])
        env.action=[.95]*len(BOUNDS)
        env.step([2]*len(BOUNDS))
        self.assertEqual(env.action,[1.]*len(BOUNDS))
        for invalid in (0,-.1,float("nan"),float("inf"),3):
            with self.assertRaises(ValueError):
                SizedReceiverEnv((TargetSpec.from_existing_thresholds(),),step_size=invalid)

    def test_checkpoint_preserves_step_size_and_legacy_default(self):
        import torch
        agent=PPOAgent(STATE_DIM,len(BOUNDS),seed=1)
        with TemporaryDirectory() as temp:
            path=Path(temp)/"policy.pt"
            save_policy(agent,path,step_size=.3)
            self.assertEqual(load_policy(path).step_size,.3)
            data=torch.load(path,weights_only=True)
            del data["step_size"]
            torch.save(data,path)
            self.assertEqual(load_policy(path).step_size,.1)

    def test_sizing_roundtrip_and_bounds(self):
        c=SimulationConditions(circuit_sizing=CircuitSizing(input_width_um=25,physical_bias=True))
        self.assertEqual(SimulationConditions(**c.to_dict()).circuit_sizing,c.circuit_sizing)
        self.assertEqual(ReceiverParameters().spice_parameters(c)["WIN"],25)
        self.assertEqual(circuit_block(c).name,"ctle_physical_bias.spice")
        with self.assertRaises(ValueError): CircuitSizing(input_length_um=.01).validate()

    def test_bias_block_contains_mirror_instead_of_ideal_tail(self):
        source=circuit_block(SimulationConditions(circuit_sizing=CircuitSizing(physical_bias=True))).read_text()
        self.assertIn("XTAIL",source); self.assertIn("XREF",source)
        self.assertNotIn("ITAIL tail",source)

    def test_prbs15_period_and_jitter_determinism(self):
        bits=generate_bits("prbs15",32767*2,7)
        self.assertEqual(bits[:32767],bits[32767:])
        self.assertNotEqual(bits[:127],bits[127:254])
        cfg=NRZStimulusConfig(jitter_rms_s=5e-12)
        a=generate_nrz(cfg); b=generate_nrz(cfg)
        np.testing.assert_array_equal(a.differential_v,b.differential_v)
        self.assertNotEqual(a.checksum,generate_nrz(replace(cfg,jitter_rms_s=0)).checksum)
        with self.assertRaises(ValueError): generate_nrz(replace(cfg,bit_rate=0))

    def test_measurements_against_known_signal(self):
        t=np.arange(0,220e-9,1e-11)
        signal=np.sin(2*np.pi*1e8*t)+.001*np.sin(2*np.pi*3e8*t)
        self.assertAlmostEqual(hd3_db(Trace("time",t,{"vout_diff":signal})), -60,places=3)
        f=np.linspace(1e7,5e9,1000); density=np.full_like(f,1e-9)
        self.assertAlmostEqual(integrated_input_noise(Trace("frequency",f,{"inoise_spectrum":density})),1e-9*np.sqrt(5e9-1e7),places=12)

    def test_budget_stops_new_evaluations(self):
        calls=[]
        def fake(p,c,f,**kwargs):
            calls.append(kwargs)
            return ReceiverEvaluation(False,p,c,f,(),{},"dc",0.,"id",{})
        with TemporaryDirectory() as temp:
            evaluator=BudgetedEvaluator(SearchLimits(10,1),cache_root=temp,evaluator=fake)
            evaluator(ReceiverParameters(),SimulationConditions(),EvaluationFidelity.TRAINING)
            with self.assertRaises(SearchBudgetExpired): evaluator(ReceiverParameters(),SimulationConditions(),EvaluationFidelity.TRAINING)
            self.assertEqual(len(calls),1); self.assertIn("deadline",calls[0])

    def test_validation_cache_hits_do_not_consume_simulation_budget(self):
        def fake(p,c,f,**kwargs):
            return ReceiverEvaluation(False,p,c,f,(),{},"dc",0.,"id",{},cache_hit=True)
        with TemporaryDirectory() as temp:
            evaluator=BudgetedEvaluator(SearchLimits(10,1),cache_root=temp,evaluator=fake,charge_cache_hits=False)
            for _ in range(3):
                evaluator(ReceiverParameters(),SimulationConditions(),EvaluationFidelity.TRAINING)
            self.assertTrue(evaluator.available())
            self.assertEqual(evaluator.calls,0)
            self.assertEqual(evaluator.cache_hits,3)

    def test_deadline_stops_without_calling_simulator(self):
        with TemporaryDirectory() as temp:
            evaluator=BudgetedEvaluator(SearchLimits(1,2),cache_root=temp)
            evaluator.deadline=0
            self.assertFalse(evaluator.available())
            self.assertEqual(evaluator.calls,0)

    def test_sizing_policy_checkpoint_schema(self):
        params,conditions=decode([0.]*len(BOUNDS))
        conditions.validate(); params.validate()
        self.assertTrue(conditions.circuit_sizing.physical_bias)
        agent=PPOAgent(STATE_DIM,len(BOUNDS),seed=3)
        with TemporaryDirectory() as temp:
            file=Path(temp)/"policy.pt"; save_policy(agent,file)
            loaded=load_policy(file)
            self.assertEqual(agent.act([0.]*STATE_DIM,deterministic=True)[0],loaded.act([0.]*STATE_DIM,deterministic=True)[0])
            import torch
            torch.save(agent.policy.state_dict(),file)
            with self.assertRaises(ValueError): load_policy(file)

    def test_area_tracks_sizing_and_remains_an_estimate(self):
        a=estimate_area(ReceiverParameters(),CircuitSizing())
        b=estimate_area(ReceiverParameters(),CircuitSizing(input_width_um=40,physical_bias=True),include_sampler=True)
        self.assertGreater(b["estimated_area_mm2"],a["estimated_area_mm2"])
        self.assertFalse(b["layout_verified"])

    def test_physical_feedback_sign_is_wired_correctly(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); model=root/"model"; model.touch(); stimulus=root/"stimulus"; stimulus.touch()
            positive=render_bench(ReceiverParameters(dfe_tap_v=.01),SimulationConditions(),model,stimulus,end_s=1e-9)
            negative=render_bench(ReceiverParameters(dfe_tap_v=-.01),SimulationConditions(),model,stimulus,end_s=1e-9)
            self.assertIn("sump sumn qb q vdd",positive)
            self.assertIn("sump sumn q qb vdd",negative)
            self.assertIn("XRX outn outp",negative)

    def test_model_bundle_preserves_includes_and_rejects_cycles(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); a=root/"a.spice"; b=root/"b.spice"
            a.write_text('.include "b.spice"\n'); b.write_text('.model demo nmos\n')
            self.assertIn('.model demo nmos',flatten_model(a))
            b.write_text('.include "a.spice"\n')
            with self.assertRaises(ValueError): flatten_model(a)

    def test_export_bundle_has_no_external_netlist_paths(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); model=root/"model.spice"; model.write_text('* fixture model\n')
            output=root/"bundle"
            export_bundle(ReceiverParameters(),output,model=model)
            self.assertTrue((output/"receiver_transient.cir").exists())
            self.assertIn('.lib "models.spice"',(output/"ctle_dc.cir").read_text())
            self.assertNotIn("@@",(output/"receiver_transient.cir").read_text())
            self.assertTrue((output/"stimulus.inc").exists())

    def test_benchmark_budget_and_strict_success(self):
        from experiments.benchmark_sized_receiver import trial
        class FakeEvaluator:
            def __init__(self): self.calls=0
            def available(self): return self.calls<3
            def summary(self): return {"elapsed_s":self.calls,"evaluation_calls":self.calls,"cache_hits":0}
            def __call__(self,p,c,f):
                self.calls+=1
                metrics={"dfe_locked_phase_eye_height_v":.5,"dfe_eye_width_ui":.8,
                         "dfe_min_margin_v":.2,"ctle_power_w":.001,"peaking_db":6.}
                return ReceiverEvaluation(True,p,c,f,(),metrics,None,0.,"id",{})
        for method in ("random","cem"):
            result=trial(method,42,TargetSpec.from_existing_thresholds(),SearchLimits(),evaluator=FakeEvaluator())
            self.assertEqual(len(result["rows"]),3)
            self.assertEqual(result["success_rate"],1.)
            self.assertEqual(result["first_success_evaluation"],1)

    def test_report_does_not_invent_unmeasured_response(self):
        from analysis.design_report import write_report
        with TemporaryDirectory() as temp:
            output=Path(temp)/"report.html"
            write_report(ReceiverParameters(),SimulationConditions(),{},output)
            self.assertIn("not retained",output.read_text())
            self.assertIn("NOT CLAIMED",output.read_text())

    def test_full_validation_preserves_custom_load_and_sizing(self):
        c=SimulationConditions(output_load_f=40e-15,circuit_sizing=CircuitSizing(physical_bias=True))
        jobs=scenarios(c,["example.s4p"],full_pvt=True)
        self.assertEqual(len(jobs),62)
        self.assertTrue(all(job[1].output_load_f==40e-15 for job in jobs))
        self.assertTrue(all(job[1].circuit_sizing.physical_bias for job in jobs))

if __name__=="__main__": unittest.main()
