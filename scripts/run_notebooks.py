"""Execute local notebooks using this Python interpreter and a temporary kernel."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

# Set before importing numerical libraries in the child kernel.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', choices=['01', '02', '03'],
                        help='Run only this notebook; default: all three.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'notebooks').glob(f'{args.notebook or "0"}*.ipynb'))
    assert len(paths) == (1 if args.notebook else 3)
    with tempfile.TemporaryDirectory(prefix='phishing-jupyter-') as directory:
        temp = Path(directory)
        kernel_dir = temp / 'kernels' / 'phishing'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python 3 (phishing)', 'language': 'python',
        }))
        os.environ['IPYTHONDIR'] = str(temp / 'ipython')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temp / 'runtime')
        for path in paths:
            print(f'RUN {path.name}', flush=True)
            notebook = nbformat.read(path, as_version=4)
            manager = KernelManager(
                kernel_name='phishing',
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp / 'kernels')]))
            client = NotebookClient(notebook, km=manager, timeout=600,
                                    resources={'metadata': {'path': str(root / 'notebooks')}})
            try:
                client.execute(cleanup_kc=True)
                nbformat.validate(notebook)
                code_cells = [c for c in notebook.cells if c.cell_type == 'code']
                assert all(c.execution_count is not None for c in code_cells)
                assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
                nbformat.write(notebook, path)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            print(f'PASS {path.name}: {len(code_cells)} code cells', flush=True)


if __name__ == '__main__':
    main()
