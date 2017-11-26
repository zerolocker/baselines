#!/usr/bin/env python
import sys, re, os, subprocess, numpy as np, string, random, time

def savedir_arg(MODEL_NAME):
  MODEL_NICKNAME = sys.argv[3]
  prefix = MODEL_NAME+'_'+MODEL_NICKNAME
  return '--no-load-on-start --save-dir SaveDir/%s_%s_%05d' % (
        prefix,time.strftime("%b%d"),random.randint(0,100000))

def create_bgrun_sh_dqnNature_noDoubleQ_model(GAME_NAME):
  sh_file_content = "" 
  for run_num in range(3):
    sh_file_content += ' '.join(['python3', '-m baselines.deepq.experiments.atari.train',
      '--env', GAME_NAME, '--no-double-q',
      # savedir_arg('dqn'), # commented because I've not yest tested if deepq/.../train.py works well with --save-dir, but saving model is not very important for now. I'll just leave it commented. Maybe test it in the future.
       '&\n'
       ]
      )
  sh_file_content += 'wait\n'
  return sh_file_content

def create_bgrun_sh_DeepqWithGaze_noDoubleQ_model(GAME_NAME):
  sh_file_content = ""
  for run_num in range(3):
    sh_file_content += ' '.join(['python3', '-m baselines.DeepqWithGaze.experiments.atari.train',
      '--env', GAME_NAME, '--no-double-q',
      savedir_arg('dqnHgaze'),
       '&\n'
       ]
      )
  sh_file_content += 'wait\n'
  return sh_file_content

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

def main(bg_run_creator_func):
    basestr="""
    # doc at : http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html
    arguments = {0}
    remote_initialdir = /scratch/cluster/zharucs/oai-baseline
    +Group ="GRAD"
    +Project ="AI_ROBOTICS"
    +ProjectDescription="ale"
    +GPUJob=true
    Universe = vanilla

    # UTCS has 18 such machine, to take a look, run 'condor_status  -constraint 'GTX1080==true' 
    Requirements=(TARGET.Cuda8== true)

    executable = /bin/bash 
    getenv = true
    output = CondorOutput/$(Cluster).out
    error = CondorOutput/$(Cluster).err
    log = CondorOutput/log.txt
    priority = 1
    Queue
    """

    def fix_wrong_game_name(name):
      name = str.capitalize(name)
      if (name.lower() == 'mspacman'):
        name = "MsPacman"
      return name

    SH_FILE_DIR =  os.path.abspath('bgrun_yard')
    if not os.path.exists(SH_FILE_DIR):
      os.makedirs(SH_FILE_DIR)
    CHOSEN = [sys.argv[1]] if sys.argv[1] != 'all' else ALL_GAME_NAMES
    for GAME_NAME in CHOSEN:
        sh_file_content = bg_run_creator_func(fix_wrong_game_name(GAME_NAME))
        print(sh_file_content)
        raw_input('\nConfirm? Ctrl-C to quit.')

        sh_filename = "%s/%s_bgrun_%s.sh" % (SH_FILE_DIR, GAME_NAME, np.random.randint(65535))
        sh_f = open(sh_filename, 'w')
        sh_f.write(sh_file_content)

        submission = basestr.format(sh_filename)
        with open('submit.condor', 'w') as f:
          f.write(submission)

        subprocess.call(['condor_submit', 'submit.condor'])

model_to_func = {
        "dqnNature_noDoubleQ": create_bgrun_sh_dqnNature_noDoubleQ_model,
        "DeepqWithGaze_noDoubleQ": create_bgrun_sh_DeepqWithGaze_noDoubleQ_model
        }

if len(sys.argv) < 4:
  print("Usage: %s <GAME_NAME|all> <MODEL_NAME> <YOUR_MODEL_NICKNAME>" % __file__)
  print("'all' means run all games:", ALL_GAME_NAMES)
  print("Supported MODEL_NAME are: " , model_to_func.keys())
  sys.exit(1)

if sys.argv[2] in model_to_func:
    print( "Job output will be directed to folder ./CondorOutput")
    if not os.path.exists("CondorOutput"):
      os.mkdir("CondorOutput")
    main(model_to_func[sys.argv[2]])
else:
    print("ERROR: Wrong model name.")
    print("Supported model names are: " , model_to_func.keys())
    sys.exit(0)

