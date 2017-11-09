#!/usr/bin/python3

"""Continue learning with an existing model."""

import deeptext
import argparse
import signal
import os


def study():
  print()

  ap = argparse.ArgumentParser()
  ap.add_argument('-modeldir',default=None)
  args = vars(ap.parse_args())

  modeldir=args["modeldir"]
  if modeldir is None:
    print("Required arg 'modeldir' is missing.")
    print()
    exit()
  else:
    if not os.path.isdir(modeldir):
      print("Specified modeldir ("+modeldir+") is not a model directory.")
      print()
      exit()
    if not os.path.isfile(os.path.join(modeldir,"properties.json")):
      print("Model properties not found in "+modeldir+".  Maybe use newmodel command?")
      print()
      exit()
    if not os.path.isfile(os.path.join(modeldir,"msgs.txt")):
      print("Training messages not found in "+modeldir+".  Maybe use newmodel command?")
      print()
      exit()
  # end if modeldir... (checking that the model looks right)
     
  print("Improving the model at "+modeldir+"...")      
  alienbrain=deeptext.cortex(modeldir)
  alienbrain.train()
# end def study



if __name__ == '__main__':
  signal.signal(signal.SIGINT,deeptext.goquietly)
  study()


