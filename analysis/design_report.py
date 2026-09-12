"""Standalone report using measured values only; no simulated plots are invented."""
import html
import json
from pathlib import Path
from dataclasses import asdict
from analysis.final_specification import build_final_specification_report
from analysis.physical_area import estimate_area

def write_report(parameters,conditions,metrics,output,*,source="evaluation",runtime=None):
    report=build_final_specification_report(design_id="selected",parameters=asdict(parameters),nominal_metrics=metrics,nominal_source=source)
    area=estimate_area(parameters,conditions.circuit_sizing)
    rows="".join("<tr>"+"".join("<td>"+html.escape(str(r[k] if r[k] is not None else "not measured"))+"</td>" for k in ("metric","measured","requirement","verdict"))+"</tr>" for r in report["rows"] if r["metric"]!="Transistor channel area (mm^2)")
    points=[(1e8,"gain_100mhz_db"),(1.25e9,"gain_1p25ghz_db"),(2.5e9,"gain_2p5ghz_db"),(5e9,"gain_5ghz_db")]
    measured=[(f,metrics[k]) for f,k in points if k in metrics]
    plot="<p>Frequency-response samples were not retained.</p>"
    if measured:
        import math
        valid=[(f,g) for f,g in measured if math.isfinite(g)]
        circles="".join(f'<circle cx="{50+500*(math.log10(f)-8)/2}" cy="{160-3*g}" r="4" fill="#285eab"/><text x="{50+500*(math.log10(f)-8)/2}" y="195" font-size="10">{f/1e9:g} GHz</text>' for f,g in valid)
        plot='<svg viewBox="0 0 650 220" role="img" aria-label="Measured gain samples"><path d="M40 10V180H630" stroke="black" fill="none"/>'+circles+'</svg><p>Measured gain samples (dB); not a reconstructed continuous response.</p>'
    data={"parameters":asdict(parameters),"conditions":conditions.to_dict(),"metrics":metrics,"area":area,"runtime":runtime}
    document='<!doctype html><meta charset="utf-8"><title>Nebula design report</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;color:#18304a}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}svg{max-width:650px}pre{white-space:pre-wrap}</style><h1>Nebula design report</h1><p>'+html.escape(source)+'</p><table><tr><th>Metric</th><th>Measured</th><th>Requirement</th><th>Status</th></tr>'+rows+'</table><h2>Measured response</h2>'+plot+'<h2>Area estimate</h2><p>'+str(area["estimated_area_mm2"])+ ' mm². Estimate only; layout not verified.</p><h2>Design and provenance</h2><pre>'+html.escape(json.dumps(data,indent=2,default=str))+'</pre>'
    Path(output).write_text(document,encoding="utf-8")
    return data
