#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import argparse, re
from collections import defaultdict
import numpy as np
import os

def main():
  parser = argparse.ArgumentParser('You must specify a (model) name for each file. ' +
          'e.g. 23456.out Dqn 23486.out DqnWithGaze ... \n' + 
    ' np.max() is used to aggregate rewards in a episode.')
  parser.add_argument('file_and_name', metavar='file_and_name_seperated_by_a_space', nargs = '+')
  args = parser.parse_args()
  assert(len(args.file_and_name)%2==0)
  n = len(args.file_and_name)/2
  files = args.file_and_name[0:n+1:2]
  names =  args.file_and_name[1:n+2:2]

  color = iter(cm.rainbow(np.linspace(0,1,n)))
  for (file, name) in zip(files, names):
    myplot(file, next(color), name)
  finalize_plot_and_show()

def myplot(file, color, name):
  EPI_REGEX = re.compile('\| episodes              \| (\d+)')
  REWARD_REGEX = re.compile('\| reward \(100 epi mean\) \| ([\d\.e\+\-]+)')

  epi_data = defaultdict(list)
  episode, reward = None, None
  for line in open(file,'r'):
    if EPI_REGEX.match(line):
      episode = int(EPI_REGEX.match(line).group(1))
    if REWARD_REGEX.match(line):
      reward = float(REWARD_REGEX.match(line).group(1))
      epi_data[episode].append(reward)

  x,y=[], []
  for (epi, rew_list) in epi_data.items():
    x.append(epi)
    y.append(np.max(rew_list))
  plt.plot(x,y, c=color, label=name)

def finalize_plot_and_show():
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  plt.legend()
  plt.show()


if __name__ == '__main__':
  main()
