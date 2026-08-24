from pathlib import Path
import subprocess as subp
from sys import exit
for file in list(Path("classes").glob("*.txt")):
	with open(file) as class_type:
		classes = [x.strip() for x in class_type.readlines()]
		procs = [subp.Popen(["python", "generate_supervised_returns.py", symb, "9", "Y"]) for symb in classes] + [subp.Popen(["python", "generate_supervised_returns.py", symb, "9", "N"]) for symb in classes] + [subp.Popen(["python", "generate_supervised_returns.py", symb, "18", "Y"]) for symb in classes] + [subp.Popen(["python", "generate_supervised_returns.py", symb, "18", "N"]) for symb in classes]
		for p in procs:
			p.wait()
		for p in procs:
			if p.returncode != 0:
				exit()