# pylint: disable-msg=too-many-function-args
# pylint: disable-msg=assignment-from-no-return
import numpy as np
import csv
from tqdm import tqdm
import time

def write_to_output_file(num_features, w):
  output_file = open('./outputs/data_100k_300_stochastic_output.tsv','w+')

  for i in range(1, num_features + 1):
    output_file.write('w{} \t'.format(i))

  output_file.write('w0 \n')

  for val in w:
    output_file.write('{}\t'.format(val.item(0)))


with open('./dataset/data_100k_300.tsv', 'r') as tsv_input_file:
  tsv_input_file = csv.reader(tsv_input_file, delimiter='\t')
  num_data_points = int(next(tsv_input_file)[0])
  num_features = int(next(tsv_input_file)[0])
  next(tsv_input_file, None)

  learning_rate = 0.0000001
  m = 1
  y = []
  x = []
  w = np.random.rand(num_features + 1, 1)

  for row in tsv_input_file:
    row.append(1.0)
    y.append([float(row[0])])
    x.append([float(x) for x in row[1:]])

  x = np.matrix(x)
  y = np.matrix(y)

  for i in range(12):
    D = np.split(x, num_data_points)
    counter = 0

    for part in D:
      y_p = y.item(counter)
      # we do part transposed because we need it as a column vector
      y_p_hat = np.matmul(w.transpose(), part.transpose()).item(0)
      counter = counter + 1

      w = w + ((learning_rate / m) * ((y_p - y_p_hat) * part.transpose()))

  write_to_output_file(num_features, w)

  loss = (1 / (2 * num_data_points)) * np.matmul(np.subtract(np.matmul(x, w), y).transpose(), np.subtract(np.matmul(x, w), y))