# pylint: disable-msg=too-many-function-args
# pylint: disable-msg=assignment-from-no-return
import numpy as np
import csv
from tqdm import tqdm
import time
import argparse

def constructArguments():
  # construct the argument parse and parse the arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-e", "--epochs", type=int, default=20,
    help="# of epochs")
  parser.add_argument("-l", "--learningRate", type=float, default=0.000001,
    help="learning rate")
  parser.add_argument("-b", "--batchSize", type=int, default=1,
    help="size of SGD batches")
  parser.add_argument("-i", "--input", type=str, default=None,
    help="input path of data file")
  parser.add_argument("-o", "--output", type=str, default=None,
    help="output path of data file")
  args = vars(parser.parse_args())

  return args


def write_to_output_file(num_features, w):
  output_file = open(args["output"], 'w+')

  for i in range(1, num_features + 1):
    output_file.write('w{} \t'.format(i))

  output_file.write('w0 \n')

  for val in w:
    output_file.write('{}\t'.format(val.item(0)))

args = constructArguments()

with open(args["input"], 'r') as tsv_input_file:
  tsv_input_file = csv.reader(tsv_input_file, delimiter='\t')
  num_data_points = int(next(tsv_input_file)[0])
  num_features = int(next(tsv_input_file)[0])
  next(tsv_input_file, None)

  learning_rate = args["learningRate"]
  m = args["batchSize"]
  epochs = args["epochs"]
  y = []
  x = []
  w = np.random.rand(num_features + 1, 1)

  for row in tsv_input_file:
    row.append(1.0)
    y.append([float(row[0])])
    x.append([float(x) for x in row[1:]])

  x = np.matrix(x)
  y = np.matrix(y)

  for i in range(epochs):
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
