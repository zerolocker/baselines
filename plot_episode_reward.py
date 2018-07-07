#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import argparse, re
from collections import defaultdict
import numpy as np
import os, subprocess, sys
from IPython import embed

aggregate_func = np.mean
aggregate_func_meaning = 'If there are >1 file becasue you used path pattern, rewards belonging to the same episode is aggregrated by: ' + aggregate_func.__name__

def main():
  parser = argparse.ArgumentParser('You must specify a model name for each file/file pattern. ' +
    'e.g. 23456.out Dqn 23486.out DqnWithGaze\n' + 
    '[path pattern is also supported]\n' +
    'e.g. SaveDir/dqnHgaze_freezeGaze*/log.txt freezeGaze SaveDir/dqnHgaze_trainableGaze*/log.txt trainableGaze \n' + aggregate_func_meaning)

  parser.add_argument('files_and_modelnames', metavar='files_and_modelnames_seperated_by_a_space', nargs = '+')
  args = parser.parse_args()
  assert len(args.files_and_modelnames)%2==0, \
     "Number of arguments is not even. Expecting pairs of (file_pattern, modelname). Got: %s" % args.files_and_modelnames \
     + "\nPlease also make sure you put quote '' around the arugment that contains '*'"
  n = len(args.files_and_modelnames)
  files_pattern = args.files_and_modelnames[0:n:2]
  modelnames =  args.files_and_modelnames[1:n:2]

  # concatenate all log files into a python list, grouped by model name
  concated_log = []
  for (i, (pattern, modelname)) in enumerate(zip(files_pattern, modelnames)):
    if modelname == '__':
      modelnames[i] = os.path.basename(pattern)
      modelname = os.path.basename(pattern)
    matched_files = subprocess.check_output("ls -R " + pattern, shell=True).decode('utf-8')
    print("The following files matched for model '%s':" % modelname)
    print(matched_files)
    matched_files = matched_files.split()
    concated_log.append([])
    for fname in matched_files:
      assert "ASCII" in subprocess.check_output(["file", fname]).decode('utf-8'), "This file is not ASCII file: " + fname + " . You should only match log.txt by using e.g. '/DQN_doubleQ_*/log.txt'"
      with open(fname, 'r') as f:
        concated_log[-1] += f.readlines() # readlines() returns a list

  print(aggregate_func_meaning)
  color = iter(cm.rainbow(np.linspace(0,1,n/2)))
  for (log, modelname) in zip(concated_log, modelnames):
    myplot(log, next(color), modelname)
    print_initial_freeze_phase_last_episode_if_exists(modelname, log)
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
    y.append(aggregate_func(rew_list))
  plt.plot(x,y, c=color, label=modelname)

def finalize_plot_and_show():
  plt.xlabel("Episode")
  plt.ylabel("Reward")
  plt.legend(loc='upper left')
  plt.show()

def print_initial_freeze_phase_last_episode_if_exists(modelname, log):
    INIT_FREEZE_ITER_REGEX = re.compile(r'\| init_freeze_iter\ +\| (\d+)')
    init_freeze_iter = None
    for line in log[:1000]: # if the "argument table" dump exists, it must be at the first 1000 line
        if init_freeze_iter is None and INIT_FREEZE_ITER_REGEX.match(line):
            init_freeze_iter = int(INIT_FREEZE_ITER_REGEX.match(line).group(1))
    if init_freeze_iter is None:
        return
    EPI_REGEX = re.compile(r'\| episodes              \| (\d+)')
    ITER_REGEX = re.compile(r'\| iters\ +\| (\d+)')
    for line in log:
        if EPI_REGEX.match(line):
            episode = int(EPI_REGEX.match(line).group(1))
        if ITER_REGEX.match(line):
            iter = int(ITER_REGEX.match(line).group(1))
            if iter > init_freeze_iter:
                print("[auxiliary info] modelname: %s iter_freeze_iter: %d initial_freeze_phase_last_episode: %d" % (modelname, init_freeze_iter, episode))
                return


if __name__ == '__main__':
  main()
