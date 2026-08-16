import subprocess
from pathlib import Path
from sys import argv
files = [str(file)[:-10] for file in Path('.').glob('*.parquet')]
for file in files:
	print(subprocess.run(['mv', file + '_9.parquet', file + '.parquet']).stdout)