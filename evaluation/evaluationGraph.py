import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Metric Evaluation")
parser.add_argument("-f", "--filename", help="Gold standard", required=True, type=str)
args = parser.parse_args()
config = vars(args)

data = open(args.filename, 'r')
lines = data.readlines()

category = list()
metrics = list()
b = ''
first = True
for line in lines:

    if "category" in line:
        a = line.split(' ')
        a[1] = a[1].strip()
        m = metrics.copy()
        if not first:
            category.append((b, m))
            metrics.clear()
        b = a[1]
        first = False

    elif len(line) > 4:
        a = line.split('\t')
        a[1] = a[1].strip()
        metrics.append(a)

for c in category:
    print(c)
x = [category[i][0]
     for i in range(0, len(category))]
#distance
y = [category[i][1][0][1]
     for i in range(0, len(category))]

x = np.array(x)
y = np.array(y)
plt.plot(x, y)
plt.show()
#f1_score
x = [category[i][1][1][1]
     for i in range(0, len(category))]

y = np.array(y)
plt.plot(x, y)
plt.show()

