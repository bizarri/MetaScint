from __future__ import annotations
import time
import itertools
import math
import numpy as np
from typing import Callable, Optional

from optimization.interfaces import (
    SearchSpace,
    NumericParameter,
    CategoricalParameter,
    SolverConfig,
    OptimizationResult,
    ObjectiveFunction,
    Constraint,
)


def _iter_grid(space: SearchSpace) -> list[dict[str, float | str]]:
    grids = []
    for p in space.parameters:
        if isinstance(p, NumericParameter):
            if p.step and p.step > 0:
                vals = np.arange(p.min, p.max + p.step * 0.5, p.step)
                vals = [float(v) for v in vals if v >= p.min - 1e-12 and v <= p.max + 1e-12]
            else:
                vals = [p.default] if p.default is not None else [p.min]
            grids.append([(p.name, v) for v in vals])
        elif isinstance(p, CategoricalParameter):
            grids.append([(p.name, c) for c in p.choices])
    if not grids:
        return [{}]
    return [dict(combo) for combo in itertools.product(*grids)]


def _sample_random(
    space: SearchSpace, n: int, rng: np.random.Generator,
) -> list[dict[str, float | str]]:
    samples = []
    for _ in range(n):
        pt = {}
        for p in space.parameters:
            if isinstance(p, NumericParameter):
                pt[p.name] = float(rng.uniform(p.min, p.max))
            elif isinstance(p, CategoricalParameter):
                pt[p.name] = str(rng.choice(p.choices))
        samples.append(pt)
    return samples


def _satisfies_constraints(constraints: list[Constraint], cand: dict) -> bool:
    return all(c.satisfied(cand) for c in constraints)


def _extract_metrics(result: dict) -> dict:
    return {
        'full_count': result.get('full_count', 0),
        'full_fraction': result.get('full_fraction', 0),
        'total_deposited_keV': result.get('total_deposited_keV', 0),
        'figure_of_merit': result.get('figure_of_merit', 0),
        'avg_detection_time_ns': result.get('avg_detection_time_ns', 0),
    }


class GridSearch:
    def __init__(
        self,
        space: SearchSpace,
        objective: ObjectiveFunction,
        evaluator: Callable,
        constraints: Optional[list[Constraint]] = None,
        config: Optional[SolverConfig] = None,
    ):
        self.space = space
        self.objective = objective
        self.evaluator = evaluator
        self.constraints = constraints or []
        self.config = config or SolverConfig(method='grid')

    def run(self) -> OptimizationResult:
        candidates = _iter_grid(self.space)
        candidates = [c for c in candidates if self._satisfies_constraints(c)]
        if not candidates:
            raise ValueError('No valid candidates after constraint filtering')
        maximize = self.objective.config.direction == 'maximize'
        best_val = -float('inf') if maximize else float('inf')
        best_cand = candidates[0]
        history: list[dict] = []
        t0 = time.time()

        for cand in candidates:
            result = self.evaluator.evaluate(cand)
            val = self.objective.evaluate(result)
            history.append({'candidate': dict(cand), 'value': float(val), **_extract_metrics(result)})
            if (maximize and val > best_val) or (not maximize and val < best_val):
                best_val = val
                best_cand = cand

        wall_time = time.time() - t0
        return OptimizationResult(
            best_candidate=best_cand,
            best_value=float(best_val),
            history=history,
            wall_time_s=wall_time,
        )

    def _satisfies_constraints(self, cand: dict) -> bool:
        return _satisfies_constraints(self.constraints, cand)


class RandomSearch:
    def __init__(
        self,
        space: SearchSpace,
        objective: ObjectiveFunction,
        evaluator: Callable,
        constraints: Optional[list[Constraint]] = None,
        config: Optional[SolverConfig] = None,
    ):
        self.space = space
        self.objective = objective
        self.evaluator = evaluator
        self.constraints = constraints or []
        self.config = config or SolverConfig(method='random', n_evaluations=50)

    def run(self) -> OptimizationResult:
        rng = np.random.default_rng(self.config.seed)
        maximize = self.objective.config.direction == 'maximize'
        best_val = -float('inf') if maximize else float('inf')
        best_cand: dict = {}
        history: list[dict] = []
        t0 = time.time()

        for _ in range(self.config.n_evaluations):
            valid = False
            for _ in range(100):
                cand = _sample_random(self.space, 1, rng)[0]
                if self._satisfies_constraints(cand):
                    valid = True
                    break
            if not valid:
                continue
            result = self.evaluator.evaluate(cand)
            val = self.objective.evaluate(result)
            history.append({'candidate': dict(cand), 'value': float(val), **_extract_metrics(result)})
            if (maximize and val > best_val) or (not maximize and val < best_val):
                best_val = val
                best_cand = cand

        wall_time = time.time() - t0
        return OptimizationResult(
            best_candidate=best_cand,
            best_value=float(best_val),
            history=history,
            wall_time_s=wall_time,
        )

    def _satisfies_constraints(self, cand: dict) -> bool:
        return _satisfies_constraints(self.constraints, cand)


class AdaptiveSearch:
    """Two-stage search: coarse grid + local refinement around best region.

    Stage 1 evaluates a coarse grid over the numeric parameters.
    Stage 2 samples randomly in a narrowed window around the best candidate.
    """
    def __init__(
        self,
        space: SearchSpace,
        objective: ObjectiveFunction,
        evaluator: Callable,
        constraints: Optional[list[Constraint]] = None,
        config: Optional[SolverConfig] = None,
    ):
        self.space = space
        self.objective = objective
        self.evaluator = evaluator
        self.constraints = constraints or []
        self.config = config or SolverConfig(method='adaptive', n_evaluations=50, n_initial=10)

    def run(self) -> OptimizationResult:
        maximize = self.objective.config.direction == 'maximize'
        best_val = -float('inf') if maximize else float('inf')
        best_cand: dict = {}
        history: list[dict] = []
        t0 = time.time()

        candidates = _iter_grid(self.space)
        candidates = [c for c in candidates if self._satisfies_constraints(c)]

        n_stage1 = min(self.config.n_initial, len(candidates))
        if n_stage1 < len(candidates):
            idx = np.linspace(0, len(candidates) - 1, n_stage1, dtype=int)
            stage1 = [candidates[i] for i in idx]
        else:
            stage1 = candidates

        for cand in stage1:
            result = self.evaluator.evaluate(cand)
            val = self.objective.evaluate(result)
            history.append({'candidate': dict(cand), 'value': float(val), **_extract_metrics(result)})
            if (maximize and val > best_val) or (not maximize and val < best_val):
                best_val = val
                best_cand = cand

        remaining = self.config.n_evaluations - len(stage1)
        if remaining > 0 and best_cand:
            seed = self.config.seed
            rng = np.random.default_rng(None if seed is None else seed + 1)
            num_params = [p for p in self.space.parameters if isinstance(p, NumericParameter)]
            for _ in range(remaining):
                cand = dict(best_cand)
                for p in num_params:
                    current = float(cand.get(p.name, (p.min + p.max) / 2))
                    span = (p.max - p.min) * 0.3
                    lo = max(p.min, current - span / 2)
                    hi = min(p.max, current + span / 2)
                    cand[p.name] = float(rng.uniform(lo, hi))
                if not self._satisfies_constraints(cand):
                    continue
                result = self.evaluator.evaluate(cand)
                val = self.objective.evaluate(result)
                history.append({'candidate': dict(cand), 'value': float(val), **_extract_metrics(result)})
                if (maximize and val > best_val) or (not maximize and val < best_val):
                    best_val = val
                    best_cand = cand

        wall_time = time.time() - t0
        return OptimizationResult(
            best_candidate=best_cand,
            best_value=float(best_val),
            history=history,
            wall_time_s=wall_time,
        )

    def _satisfies_constraints(self, cand: dict) -> bool:
        return _satisfies_constraints(self.constraints, cand)


SEARCH_REGISTRY: dict[str, type] = {
    'grid': GridSearch,
    'random': RandomSearch,
    'adaptive': AdaptiveSearch,
}


def build_solver(
    method: str,
    space: SearchSpace,
    objective: ObjectiveFunction,
    evaluator: Callable,
    constraints: Optional[list[Constraint]] = None,
    config: Optional[SolverConfig] = None,
):
    cls = SEARCH_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {list(SEARCH_REGISTRY.keys())}"
        )
    return cls(space, objective, evaluator, constraints, config)
