import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import json
style.use("seaborn-whitegrid")

plt.figure(figsize=(6, 3.5))

n_bins = 100
xs = np.linspace(0, 1, n_bins)

for f in glob.glob("*.json"):
    x = json.load(open(f))

    eps = x["eps"]
    U = x["U"]
    p = x["p"]

    if eps == 0:
        plt.plot(xs, p, label=f"Slater point")

for f in glob.glob("*.json"):
    x = json.load(open(f))
    eps = x["eps"]
    U = x["U"]
    p = x["p"]

    if U>0:
        print(f'{U} {eps} {x["count"]} {x["Fy"]:.3f} {x["time"]:.3f}')

    if eps == 0.001:
        plt.plot(xs, p, label=f"eps={eps} U=[{-U},{U}]")

plt.xlim(0, 1)
plt.ylim(0.6, 1.6)
plt.legend()
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("1.png")
plt.show()
