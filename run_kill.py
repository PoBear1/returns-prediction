import subprocess
values = [1326, 1317, 1320, 1307, 1319, 1310, 1313, 1316, 1333, 1309, 1315, 1308, 1332, 1312, 1311, 1343, 1345]
for x in values:
	subprocess.run(["kill", str(x)])