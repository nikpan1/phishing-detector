"""Execute the solution notebook with this interpreter; save embedded outputs."""
import json
import os
from pathlib import Path
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    src = Path(__file__).resolve().parent
    path = src / 'notebooks' / 'training.ipynb'
    with tempfile.TemporaryDirectory(prefix='phishing-solution-') as directory:
        temp = Path(directory)
        kernel = temp / 'kernels' / 'solution'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv':[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name':'Python 3 (solution)', 'language':'python'}))
        os.environ['IPYTHONDIR'] = str(temp / 'ipython')
        os.environ['MPLCONFIGDIR'] = str(temp / 'matplotlib')
        os.environ['XDG_CACHE_HOME'] = str(temp / 'cache')
        Path(os.environ['XDG_CACHE_HOME']).mkdir()
        manager = KernelManager(kernel_name='solution', kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[str(temp / 'kernels')]))
        notebook = nbformat.read(path, 4)
        try:
            NotebookClient(notebook, km=manager, timeout=1200,
                           resources={'metadata':{'path':str(src)}}).execute(cleanup_kc=True)
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    print(f'PASS: {path.name}')


if __name__ == '__main__':
    main()
