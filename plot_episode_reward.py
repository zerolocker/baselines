#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import argparse, re
from collections import defaultdict
import numpy as np
import os, subprocess, sys

def main():
  parser = argparse.ArgumentParser('You must specify a model name for each file/file pattern. ' +
    'e.g. 23456.out Dqn 23486.out DqnWithGaze\n' + 
    '[path pattern is supported, in `grep` syntax]\n' +
    'e.g. dqnHgaze_freezeGaze.*/log.txt freezeGaze dqnHgaze_trainableGaze.*/log.txt trainableGaze \n' +
    'If there are >1 file becasue you used path pattern, rewards belonging to the same episode is aggregrated by np.mean().')
  parser.add_argument('files_and_modelnames', metavar='files_and_modelnames_seperated_by_a_space', nargs = '+')
  args = parser.parse_args()
  assert(len(args.files_and_modelnames)%2==0)
  n = len(args.files_and_modelnames)
  files_pattern = args.files_and_modelnames[0:n:2]
  modelnames =  args.files_and_modelnames[1:n:2]

  # concatenate all log files into a python list, grouped by model name
  concated_log = []
  for (pattern, modelname) in zip(files_pattern, modelnames):
    if (os.path.exists(pattern)):
       print("Intepreting %s as a path instead of a regex path pattern, because this file exists" % pattern)
       matched_files = pattern
    else:
      try:
        matched_files = subprocess.check_output("find . | grep '%s'" % pattern, shell=True).decode('utf-8')
      except subprocess.CalledProcessError as ex:
        print(ex)
        print("Regex might be wrong (for example, did you use * instead of .* ?)")
        sys.exit(1)
    print("modelname: '%s' pattern: '%s' matched the following files:" % (modelname, pattern))
    print(matched_files)
    matched_files = matched_files.split()
    concated_log.append([])
    for fname in matched_files:
      assert "ASCII" in subprocess.check_output(["file", fname]).decode('utf-8'), "This file is not ASCII file: " + fname + " . Did you use regex to match any file? You should only match log.txt by using e.g. '/model_.*/log.txt'"
      with open(fname, 'r') as f:
        concated_log[-1] += f.readlines() # readlines() returns a list

  color = iter(cm.rainbow(np.linspace(0,1,n/2)))
  for (log, modelname) in zip(concated_log, modelnames):
    myplot(log, next(color), modelname)
  finalize_plot_and_show()

def myplot(log, color, modelname):
  EPI_REGEX = re.compile(r'\| episodes              \| (\d+)')
  REWARD_REGEX = re.compile(r'\| reward \(100 epi mean\) \| ([\d\.e\+\-]+)')

  epi_data = defaultdict(list)
  episode, reward = None, None
  for line in log:
    if EPI_REGEX.match(line):
      episode = int(EPI_REGEX.match(line).group(1))
    if REWARD_REGEX.match(line):
      reward = float(REWARD_REGEX.match(line).group(1))
      epi_data[episode].append(reward)

  x,y=[], []
  for (epi, rew_list) in epi_data.items():
    x.append(epi)
    y.append(np.mean(rew_list))
  plt.plot(x,y, c=color, label=modelname)

def finalize_plot_and_show():
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  plt.legend(loc='upper left')
  plt.show()


if __name__ == '__main__':
  main()
