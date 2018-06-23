#!/usr/bin/env python
import sys, re, os, subprocess, numpy as np, string, random, time

def save_model_args(model_name, randint):
  prefix = model_name+'_'+EXPR_NAME
  return '--resumable --save-dir SaveDir/%s_%s_%05d' % (
        prefix,time.strftime("%b%d"), randint)

def create_bgrun_sh_dqnNature_noDoubleQ_model(GAME_NAME, randint):
  sh_file_content = "" 
  for run_num in range(3):
    sh_file_content += ' '.join(['python3', '-m baselines.deepq.experiments.atari.train',
      '--env', GAME_NAME, '--no-double-q',
      # save_model_args('dqn'), # commented because I've not yest tested if deepq/.../train.py works well with --save-dir, but saving model is not very important for now. I'll just leave it commented. Maybe test it in the future.
       ] + OTHER_PARAMETERS_TO_PASS)
       
    sh_file_content += ' &\n'
  sh_file_content += 'wait\n'
  return sh_file_content

def create_bgrun_sh_DeepqWithGaze_noDoubleQ_model(GAME_NAME, randint):
  sh_file_content = ""
  for run_num in range(1):
    sh_file_content += ' '.join(['python3', '-m baselines.DeepqWithGaze.experiments.atari.train',
      '--env', GAME_NAME, '--no-double-q',
      save_model_args('dqnHgaze', randint),
      ] + OTHER_PARAMETERS_TO_PASS)
    sh_file_content += ' &\n'
  sh_file_content += 'wait\n'
  return sh_file_content

def fix_wrong_game_name(name):
  name = str.capitalize(name)
  if (name.lower() == 'mspacman'):
    name = "MsPacman"
  return name

ALL_GAME_NAMES=[
   ("Breakout"),
   ("Centipede"),
   ("Enduro"),
   ("Freeway"),
   ("MsPacman"),
   ("Riverraid"),
   ("Seaquest"),
   ("Venture"),
]

MODEL_SH_MAPPING = {
        "dqnNature_noDoubleQ": create_bgrun_sh_dqnNature_noDoubleQ_model,
        "DeepqWithGaze_noDoubleQ": create_bgrun_sh_DeepqWithGaze_noDoubleQ_model,
        }

if len(sys.argv) < 4:
    print("Usage: %s <GAME_NAME|all> <MODEL_NAME> <YOUR_EXPR_NAME> <OTHER_PARAMETERS_TO_PASS>" % __file__)
    print("'all' means run all games:", ALL_GAME_NAMES)
    print("Supported MODEL_NAME are: " , MODEL_SH_MAPPING.keys())
    sys.exit(1)

OTHER_PARAMETERS_TO_PASS = sys.argv[4:] # TODO: refactor this and be less hacky
EXPR_NAME = sys.argv[3]
CHOSEN = [sys.argv[1]] if sys.argv[1] != 'all' else ALL_GAME_NAMES
SH_FILE_DIR =  os.path.abspath('bgrun_yard')

if sys.argv[2] not in MODEL_SH_MAPPING:
    print("ERROR: Wrong model name.")
    print("Supported model names are: " , MODEL_SH_MAPPING.keys())
    sys.exit(0)
else:
    bg_run_creator_func = MODEL_SH_MAPPING[sys.argv[2]]
    print( "Job output will be directed to folder ./CondorOutput")
    if not os.path.exists("CondorOutput"):
      os.mkdir("CondorOutput")
    
    basestr="""
    # doc at : http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html
    arguments = {0}
    remote_initialdir = {1}
    +Group ="GRAD"
    +Project ="AI_ROBOTICS"
    +ProjectDescription="ale"
    +GPUJob=true
    Universe = vanilla

    # UTCS has 18 such machine, to take a look, run 'condor_status  -constraint 'GTX1080==true' 
    Requirements=(TARGET.Cuda8== true)

    executable = /bin/bash 
    getenv = true
    output = CondorOutput/{2}.$(Cluster).out
    error = CondorOutput/{2}.$(Cluster).err
    log = CondorOutput/log.txt
    priority = 1
    Queue
    """

    if not os.path.exists(SH_FILE_DIR):
      os.makedirs(SH_FILE_DIR)
    for GAME_NAME in CHOSEN:
        randint = np.random.randint(65535)
        sh_file_content = bg_run_creator_func(fix_wrong_game_name(GAME_NAME), randint)
        print(sh_file_content)
        raw_input('\nConfirm? Ctrl-C to quit.')

        sh_filename = "%s/%s_%s_bgrun_%s.sh" % (SH_FILE_DIR, EXPR_NAME, GAME_NAME, randint)
        sh_f = open(sh_filename, 'w')
        sh_f.write(sh_file_content)

        with open('submit.condor', 'w') as f:
          f.write(basestr.format(sh_filename, os.path.abspath('.'), EXPR_NAME))

        subprocess.call(['condor_submit', 'submit.condor'])

