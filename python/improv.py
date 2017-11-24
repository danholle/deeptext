#!/usr/bin/python3

"""Continue learning with an existing model."""

import deeptext
import argparse
import os
import random

def improv():
  print()

  ap = argparse.ArgumentParser()
  ap.add_argument('-modeldir',default=None)
  ap.add_argument('-primetext',default=None)
  ap.add_argument('-repeats',type=int,default=5)
  ap.add_argument('-temperature',type=float,default=0.4)
  ap.add_argument('-pratio',type=float,default=0.1)
  args = vars(ap.parse_args())

  repeats=args["repeats"]
  pratio=args["pratio"]
  temperature=args["temperature"]

  modeldir=args["modeldir"]
  if modeldir is None:
    print("Required arg 'modeldir' is missing.")
    print("Models are typically in the models directory:  look there and try something")
    print("like '-modeldir models/quotes'.")
    print()
    exit()
  else:
    if not os.path.isdir(modeldir):
      print("Specified modeldir ("+modeldir+") is not a model directory.")
      print()
      exit()
    if not os.path.isfile(os.path.join(modeldir,"properties.json")):
      print("Model properties not found in "+modeldir+".  Maybe use newmodel command first?")
      print()
      exit()
    if not os.path.isfile(os.path.join(modeldir,"msgs.txt")):
      print("Training messages not found in "+modeldir+".  Maybe use newmodel command first?")
      print()
      exit()
  # end if modeldir... (checking that the model looks right)

  alienbrain=deeptext.cortex(modeldir)

  primetext=args["primetext"]
  if primetext is None:
    print("Improvising how some "+alienbrain.label+" might have ended differently:")
    print()
    print("="*50)
    print()
    for i in range(repeats):
      origmsg=alienbrain.msgs[random.randint(0,len(alienbrain.msgs)-1)]
      print("Original:")
      deeptext.wrapsody(" -- ",origmsg)
      prefix=origmsg[:len(origmsg)//2]
      print()
      print("Improvised by an alien brain:")
      for j in range(repeats):
        deeptext.wrapsody(" -- ",alienbrain.finishmsg(prefix,
            temperature=temperature,pratio=pratio))
      print()
      print("="*50)
      print()
  elif primetext=="":
    print("Improvising some random "+alienbrain.label+":")  
    print()
    for _ in range(repeats):
      deeptext.wrapsody(" -- ",alienbrain.finishmsg("",
          temperature=temperature,pratio=pratio)) 
    print()
  else:
    print("Improvising "+alienbrain.label+" starting with \""+primetext+"\":")
    print()
    for _ in range(repeats):
      deeptext.wrapsody(" -- ",alienbrain.finishmsg(primetext,
          temperature=temperature,pratio=pratio))
    print()
  # end if / breaking out by type of improv

# end def improv


if __name__ == '__main__':
  improv()


