"""Post-hoc analysis of existing NEBULA optimization-baseline results.

Everything in this package reads already-written results/*.jsonl files. It
never imports simulator.receiver or simulator.rl_adapter's evaluator, never
calls ngspice, and never runs a new search/training loop.
"""
