from _bootstrap import ROOT
import subprocess, sys
for script in ['run_uniform_benchmark.py', 'run_composite_benchmark.py', 'run_geometry_validation.py', 'run_ray_trace.py']:
    print(f'--- Running {script} ---')
    subprocess.run([sys.executable, str(ROOT / 'workflows' / script)], check=True)
