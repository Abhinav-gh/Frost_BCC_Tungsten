"""Frost's runtime: one certified multirate executor for every neural simulator it runs.

The same five pieces drive molecular dynamics (PyTorch), our CFD and MHD closures (JAX) and
NeuralGCM (JAX):

- ``reuse``: what a runtime may hold, decided by how the consumer uses the tensor
  (structural, solver-, operator-, value- or gradient-consumed);
- ``clauses``: the certificate as plug-ins with two roles: priors that bound the interval
  before it runs (resonance), and checks on each closed interval that either roll it back
  (magnitude, non-finite, hold error) or refuse it for good (drift, regime envelope); the
  approximate-computing monitors the paper compares against (Green, Rumba) are clauses too;
- ``executor``: the loop. It draws an interval, asks the backend to run it in the chosen mode
  (sync, lag, spec), checks it, rolls back or latches a refusal, and logs every decision;
- ``planner``: picks the mode and interval from profiled costs and the clauses' priors;
- backends: ``md`` (a student every step, the teacher's residual refreshed every k steps)
  and ``hold`` (a learned closure refreshed every k steps of a classical solver).

A new workload needs a backend: four methods (``interval``, ``checkpoint``, ``restore``,
``time``), and a declaration of its operator's reuse class.
"""

from frost.runtime.clauses import (  # noqa: F401
    Clause, DriftPosterior, HoldError, LocalMagnitude, Magnitude, NonFinite, QoSThreshold, RegimeEnvelope,
    KineticBias, ResonancePrior, StructureClause, ThermostatBias, WorstCase,
)
from frost.runtime.executor import Executor, Interval  # noqa: F401
from frost.runtime.reuse import ReuseClass, check_hold  # noqa: F401
