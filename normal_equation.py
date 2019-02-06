import numpy as np

lines = [line.rstrip('\n') for line in open('./sample_input.tsv')]

num_data_points = int(lines[0])
num_features = int(lines[1])

for i in range(3):
  lines.pop(0)

x = [i.split("\t") for i in lines]
x = [[int(float(j)) for j in i] for i in x]
y = [[int(float(i[0]))] for i in x]

for list in x:
  list.pop(0)
  list.append(1)

x = np.matrix(x)
y = np.matrix(y)
w = np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), np.matmul(x.transpose(), y))
