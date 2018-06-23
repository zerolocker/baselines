#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import argparse, re
from collections import defaultdict
import numpy as np
import os, subprocess, sys
from IPython import embed

def main():
  parser = argparse.ArgumentParser('You must specify a model name for each file/file pattern. ' +
    'e.g. 23456.out Dqn 23486.out DqnWithGaze')
  parser.add_argument('files_and_modelnames', metavar='files_and_modelnames_seperated_by_a_space', nargs = '+')
  args = parser.parse_args()
  assert len(args.files_and_modelnames)%2==0, \
     "Number of arguments is not even. Expecting pairs of (file_pattern, modelname). Got: %s" % args.files_and_modelnames
  n = len(args.files_and_modelnames)
  files = args.files_and_modelnames[0:n:2]
  modelnames =  args.files_and_modelnames[1:n:2]

  logs = []
  for (fname, modelname) in zip(files, modelnames):
    logs.append([])
    assert "ASCII" in subprocess.check_output(["file", fname]).decode('utf-8'), "This file is not ASCII file: " + fname
    with open(fname, 'r') as f:
      logs[-1] += f.readlines() # readlines() returns a list

  color_iterator = iter(cm.rainbow(np.linspace(0,1,3*n/2)))
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  for (log, modelname) in zip(logs, modelnames):
    myplot(log, color_iterator, modelname, ax1, ax2)
  finalize_plot_and_show(ax1, ax2)

def myplot(log, color_iterator, modelname, reward_ax, norm_ax):
  EPI_REGEX = re.compile(r'\| episodes              \| (\d+)')
  REWARD_REGEX = re.compile(r'\| reward \(100 epi mean\) \| ([\d\.e\+\-]+)')
  W_NORM_REGEX = re.compile(r'([+-]?[\d\.e]+) ([+-]?[\d\.e\+\-]+)')

  epi_reward = dict()
  epi_norm = defaultdict(dict)
  episode, reward = None, None
  for line in log:
    if EPI_REGEX.match(line):
      episode = int(EPI_REGEX.match(line).group(1))
    if REWARD_REGEX.match(line):
      epi_reward[episode]=float(REWARD_REGEX.match(line).group(1))
    if W_NORM_REGEX.match(line):
      gaze_W_norm, qfunc_W_norm = W_NORM_REGEX.match(line).groups()
      epi_norm[episode]['gaze']= float(gaze_W_norm)
      epi_norm[episode]['qfunc']= float(qfunc_W_norm)

  x_rewardplot,y_reward, x_normplot,y_gaze_norm,y_qfunc_norm = [], [], [], [], []
  for (epi, rew) in epi_reward.items():
    x_rewardplot.append(epi)
    y_reward.append(rew)
  for (epi, norm) in sorted(epi_norm.items()):
    x_normplot.append(epi)
    y_gaze_norm.append(norm['gaze'])
    y_qfunc_norm.append(norm['qfunc'])

  reward_ax.plot(x_rewardplot,y_reward, c=next(color_iterator), label=modelname)
  norm_ax.plot(x_normplot, y_gaze_norm, c=next(color_iterator), label=modelname+' gaze_W_norm')
  norm_ax.plot(x_normplot, y_qfunc_norm, c=next(color_iterator), label=modelname+' qfunc_W_norm')

def finalize_plot_and_show(ax1, ax2):
  ax1.set_xlabel("Episode")
  ax1.set_ylabel("Reward")
  ax1.legend(loc='lower center')
  ax2.legend(loc='upper left')
  plt.show()


if __name__ == '__main__':
  main()
