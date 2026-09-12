"""Budgeted tuning characterization; a frequency is covered only by measured local peaks."""
from dataclasses import replace
from simulator.receiver import EvaluationFidelity
from simulator.runtime import BudgetedEvaluator, SearchLimits

def characterize_tuning(parameters,conditions,*, evaluator=None, frequencies=(1.25e9,1.875e9,2.5e9), tolerance=.1):
    evaluator=evaluator or BudgetedEvaluator(SearchLimits(120,9,20))
    rows=[]
    for rscale in (.5,1.,2.):
        for cscale in (.5,1.,2.):
            if not evaluator.available(): break
            candidate=replace(parameters,rdeg_ohm=parameters.rdeg_ohm*rscale,cdeg_f=parameters.cdeg_f*cscale)
            result=evaluator(candidate,conditions,EvaluationFidelity.SCREENING)
            rows.append({"rdeg_ohm":candidate.rdeg_ohm,"cdeg_f":candidate.cdeg_f,
                         "success":result.success,"metrics":result.metrics})
    coverage={str(f):any(r["success"] and r["metrics"].get("is_local_peak")==1 and
                        abs(r["metrics"].get("peak_frequency_hz",0)-f)<=tolerance*f
                        for r in rows) for f in frequencies}
    return {"sampled_frequency_coverage":coverage,"all_sampled_targets_covered":all(coverage.values()),
            "continuous_tunability_proven":False,"rows":rows,"runtime":evaluator.summary()}
