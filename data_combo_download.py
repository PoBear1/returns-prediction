from pathlib import Path
import subprocess as subp
from sys import exit
symbols = []
for file in list(Path("class").glob("*.txt")):
	with open(file) as class_type:
		symbols += [x.strip() for x in class_type.readlines()]
print(len(symbols))
num_at_once: int = 5
while len(symbols) != 0:
	classes = symbols[0: num_at_once]
	symbols = symbols[num_at_once:]
	procs = {symb: subp.Popen(["caffeinate", "-i", "-d", "python", "data_download.py", symb]) for symb in classes}
	for p in procs.values():
		p.wait()
	for symb, p in procs.items():
		if p.returncode != 0:
			symbols.append(symb)
	print(len(symbols))