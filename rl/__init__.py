"""ML-side AutoCkt-style RL implementation for the NEBULA receiver.

This package is intentionally isolated from simulator/: nothing here is
imported by simulator/*.py, and nothing in this package modifies
simulator/rl_adapter.py, simulator/receiver.py, simulator/ngspice.py,
simulator/channel.py, or simulator/cache.py. Those files are only ever
imported read-only (e.g. ACTION_BOUNDS, METRIC_OBSERVATION_NAMES,
ReceiverRLAdapter) to build a faithful AutoCkt-style state/action/reward on
top of the existing, unmodified simulator contract.

See docs/autockt-mapping.md for the full AutoCkt-source-verification and
NEBULA-adaptation design document, including which parts of this package are
[AUTOCKT-REPLICATED], [NEBULA ADAPTATION], [ROUGH-SCALE ADAPTATION], or
[UNVERIFIED-FROM-SOURCE].
"""
