from _bootstrap import ROOT
import json
import sys
from pathlib import Path

from optimization.interfaces import (
    SearchSpace, NumericParameter, CategoricalParameter,
    ObjectiveConfig, SolverConfig, ConstraintConfig,
)
from optimization.objectives import build_objective
from optimization.constraints import build_constraint
from optimization.search import build_solver
from optimization.evaluator import OptimizationEvaluator


def load_optimization_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else (
        ROOT / 'configs' / 'optimization' / 'design_search_template.json'
    )
    opt_cfg = load_optimization_config(config_path)
    study_name = opt_cfg.get('study_name', 'optimization')
    print(f'=== Optimization: {study_name} ===')
    print(f'Description: {opt_cfg.get("description", "(none)")}')
    print(f'Base config: {opt_cfg["base_config"]}')
    print(f'Events per evaluation: {opt_cfg.get("n_events", 5000)}')
    print()

    space_raw = opt_cfg.get('search_space', {})
    params: list = []
    for np_def in space_raw.get('numeric_parameters', []):
        params.append(NumericParameter(**np_def))
    for cp_def in space_raw.get('categorical_parameters', []):
        params.append(CategoricalParameter(**cp_def))
    space = SearchSpace(parameters=params)

    obj_cfg = ObjectiveConfig(**opt_cfg['objective'])
    objective = build_objective(obj_cfg)
    print(f'Objective: {objective}')
    print()

    constraints = []
    for cc_def in opt_cfg.get('constraints', []):
        constraints.append(build_constraint(ConstraintConfig(**cc_def)))
    if constraints:
        print(f'Constraints: {len(constraints)}')
        for c in constraints:
            print(f'  {type(c).__name__}')
        print()

    solver_cfg = SolverConfig(**opt_cfg.get('solver', {'method': 'grid'}))
    print(f'Solver: {solver_cfg.method}')
    print(f'  evaluations: {solver_cfg.n_evaluations}')
    print(f'  initial: {solver_cfg.n_initial}')
    print()

    n_eval = opt_cfg.get('n_events', 5000)
    base_path = ROOT / opt_cfg['base_config']
    evaluator = OptimizationEvaluator(
        base_config_path=base_path,
        n_events=n_eval,
        seed=opt_cfg.get('seed', 42),
        verbose=True,
    )

    solver = build_solver(
        method=solver_cfg.method,
        space=space,
        objective=objective,
        evaluator=evaluator,
        constraints=constraints,
        config=solver_cfg,
    )

    print('Running...')
    result = solver.run()

    print()
    print('=== Results ===')
    print(result.summary())
    print()

    best_metrics = {}
    if result.history:
        best_entry = next(
            (h for h in result.history if h['candidate'] == result.best_candidate),
            result.history[-1],
        )
        for k, v in best_entry.items():
            if k not in ('candidate', 'value'):
                best_metrics[k] = v

    output = {
        'study_name': study_name,
        'objective': obj_cfg.name,
        'direction': obj_cfg.direction,
        'solver': solver_cfg.method,
        'n_evaluations': len(result.history),
        'n_events_per_eval': n_eval,
        'best_value': float(result.best_value),
        'best_candidate': {k: float(v) if not isinstance(v, str) else v for k, v in result.best_candidate.items()},
        'best_metrics': best_metrics,
        'wall_time_s': round(result.wall_time_s, 1),
        'full_history': [
            {k: v for k, v in h.items() if k != 'candidate'} | {'candidate': dict(h['candidate'])}
            for h in result.history
        ],
    }

    out_dir = ROOT / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'optimization_{study_name}.json'
    out_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
