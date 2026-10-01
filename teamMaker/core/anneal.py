"""Generic simulated annealing (also usable by other problems, e.g. groups).

A problem object must provide:

- ``energy``: float, energy of the current state
- ``propose(rng)``: apply a random move in place and return the new energy, or
  None when no valid move was found (state unchanged)
- ``accept()``: keep the proposed move
- ``reject()``: undo the proposed move
- ``snapshot()``: return an independent copy of the current state
- ``exact_energy()``: recompute the energy of the current state from scratch
- ``at_target()``: True when the state is good enough to stop early
"""

import math
import time
from dataclasses import dataclass
from typing import Any

STOP_ITERATIONS = "iterations"
STOP_TARGET = "target"
STOP_TIME = "time"


@dataclass
class AnnealResult:
    """Outcome of one annealing run."""

    best_energy: float
    best_state: Any
    iterations: int
    accepted: int
    stopped_by: str
    elapsed: float
    t0: float
    t_end: float


def calibrate_temperatures(
    problem, rng, samples=2000, start_acceptance=0.8, end_delta=0.05, end_acceptance=0.001
):
    """Derive a start and end temperature from sampled uphill moves.

    T0 is chosen so the mean uphill delta is accepted with probability
    ``start_acceptance``; Tend so a delta of ``end_delta`` is accepted with
    probability ``end_acceptance``. The problem state is left unchanged.

    Args:
        problem: Problem object (see module docstring).
        rng: random.Random instance.
        samples: Number of moves to sample.
        start_acceptance: Acceptance probability at T0 for the mean uphill delta.
        end_delta: Reference delta for the final temperature.
        end_acceptance: Acceptance probability of ``end_delta`` at Tend.

    Returns:
        Tuple (t0, t_end) with t0 > t_end > 0.
    """
    base = problem.energy
    uphill = []
    for _ in range(samples):
        new = problem.propose(rng)
        if new is None:
            continue
        problem.reject()
        if new > base:
            uphill.append(new - base)
    t_end = -end_delta / math.log(end_acceptance)
    if uphill:
        t0 = -(sum(uphill) / len(uphill)) / math.log(start_acceptance)
    else:
        t0 = 1.0
    t0 = max(t0, t_end * 10.0)
    return t0, t_end


def anneal(
    problem,
    rng,
    iterations,
    t0,
    t_end,
    time_limit_s=None,
    check_every=1024,
):
    """Run simulated annealing with geometric cooling.

    Cooling is a function of the iteration count only, so runs bounded by
    iterations are reproducible for a given rng. A snapshot is taken on every
    strict improvement and its energy is recomputed exactly.

    Args:
        problem: Problem object (see module docstring).
        rng: random.Random instance.
        iterations: Iteration budget.
        t0: Start temperature.
        t_end: End temperature.
        time_limit_s: Optional wall-clock limit in seconds.
        check_every: How often (iterations) the clock is checked.

    Returns:
        AnnealResult with the best snapshot found.
    """
    started = time.time()
    exp = math.exp
    rand = rng.random
    propose = problem.propose
    accept = problem.accept
    reject = problem.reject

    cooling = (t_end / t0) ** (1.0 / max(1, iterations))
    temp = t0
    current = problem.energy
    best = problem.exact_energy()
    best_state = problem.snapshot()
    accepted = 0
    stopped_by = STOP_ITERATIONS
    done = 0

    if problem.at_target():
        stopped_by = STOP_TARGET
        iterations = 0

    for i in range(iterations):
        done = i + 1
        new = propose(rng)
        if new is not None:
            delta = new - current
            if delta <= 0 or rand() < exp(-delta / temp):
                accept()
                current = new
                accepted += 1
                if current < best - 1e-9:
                    best_state = problem.snapshot()
                    best = problem.exact_energy()
                    current = best
                    if problem.at_target():
                        stopped_by = STOP_TARGET
                        break
            else:
                reject()
        temp *= cooling
        if time_limit_s is not None and done % check_every == 0:
            if time.time() - started >= time_limit_s:
                stopped_by = STOP_TIME
                break

    return AnnealResult(
        best_energy=best,
        best_state=best_state,
        iterations=done,
        accepted=accepted,
        stopped_by=stopped_by,
        elapsed=time.time() - started,
        t0=t0,
        t_end=t_end,
    )
