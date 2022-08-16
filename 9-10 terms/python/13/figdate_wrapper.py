import venv
from tempfile import mkdtemp, TemporaryDirectory
import os.path as path
import subprocess
import sys


if __name__ == '__main__':
    with TemporaryDirectory() as dtemp:
        venv.create(dtemp, with_pip=True)
        subprocess.run([
            path.join(dtemp, 'bin/pip'), 'install', 'pyfiglet'
        ])
        if len(sys.argv) > 1:
            args = sys.argv[1:]
        else:
            args = []
        subprocess.run(
            [
                path.join(dtemp, 'bin/python'), '-m',
                'figdate'
            ] + args
        )
