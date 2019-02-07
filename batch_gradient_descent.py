import numpy as np
import csv

with open('./dataset/data_10k_100.tsv', 'r') as tsv_input_file:
  tsv_input_file = csv.reader(tsv_input_file, delimiter='\t')
  num_data_points = int(next(tsv_input_file)[0])
  num_features = int(next(tsv_input_file)[0])
  next(tsv_input_file, None)

  learning_rate = 0.000001
  y = []
  x = []
  w = np.random.rand(num_features + 1, 1)

  for row in tsv_input_file:
    row.append(1.0)
    y.append([float(row[0])])
    x.append([float(x) for x in row[1:]])

  x = np.matrix(x)
  y = np.matrix(y)

  x_transposed_times_y = np.matmul(x.transpose(), y)

  for i in range(200):
    x_transposed_times_x_times_w = np.matmul(np.matmul(x.transpose(), x), w)
    w = w - ((learning_rate/num_data_points) * (np.subtract(x_transposed_times_x_times_w, x_transposed_times_y)))

  loss = (1 / (2 * num_data_points)) * np.matmul(np.subtract(np.matmul(x, w), y).transpose(), np.subtract(np.matmul(x, w), y))