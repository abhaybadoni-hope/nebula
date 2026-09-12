"""Cold-cache transient speed/accuracy experiment; never changes production defaults."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from simulator.config import SimulationConditions, Sky130Config
from simulator.receiver import ReceiverParameters, EvaluationFidelity, SYNTHETIC_CHANNEL, _run_transient
from simulator.ngspice import NgSpiceConfig


def run_batch(designs, bits, workers=1):
    model=Sky130Config().resolve_model_library()
    def evaluate(item):
        name,parameters=item
        start=time.perf_counter()
        result=_run_transient(parameters,SimulationConditions(training_bit_count=bits),model,
                              NgSpiceConfig(timeout_s=30),EvaluationFidelity.TRAINING,SYNTHETIC_CHANNEL)
        return {"design":name,"elapsed_s":time.perf_counter()-start,"result":asdict(result)}
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(evaluate,designs))
    return {"bits":bits,"workers":workers,"elapsed_s":time.perf_counter()-start,"rows":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    base=ReceiverParameters(**json.loads(args.design.read_text())["parameters"])
    designs=[("baseline",base),("lower_bias",replace(base,itail_a=base.itail_a*.7)),
             ("higher_degeneration",replace(base,rdeg_ohm=base.rdeg_ohm*1.5)),
             ("large_feedback",replace(base,dfe_tap_v=.3))]
    report={"scope":"four local perturbations, nominal synthetic channel; no policy-quality claim", "batches":[]}
    for bits,workers in ((128,1),(96,1),(128,2),(96,2)):
        report["batches"].append(run_batch(designs,bits,workers))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,default=str))
        print(json.dumps({k:v for k,v in report["batches"][-1].items() if k!="rows"}),flush=True)

if __name__=="__main__": main()
