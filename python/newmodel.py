#!/usr/bin/python3

"""Create LSTM model from a message collection."""

# TODO compute defaults for hidden, intlv
# TODO something to avoid overwriting an existing model


import os
import argparse
import deeptext
import unidecode 
import signal

def newmodel():
  print()

  ap = argparse.ArgumentParser()
  ap.add_argument('-msgsfn', default=None)
  ap.add_argument('-hidden', type=int,default=0)
  ap.add_argument('-seqlen', type=int,default=60)
  ap.add_argument('-minmsglen',type=int,default=0)
  ap.add_argument('-maxmsglen',type=int,default=0)
  ap.add_argument('-label',default=None)
  ap.add_argument('-description',default="No description provided")
  ap.add_argument('-modeldir',default=None)
  args = vars(ap.parse_args())

  modeldir=args["modeldir"]
  if modeldir is None:
    print("Required arg 'modeldir' is missing.")
    exit()
  print("Model directory is "+modeldir+".")
  if (os.path.isdir(modeldir) and 
      os.path.isfile(os.path.join(modeldir,"weights.hdf5"))):
    print("...um, you know there's already a model there, right?")
    print("If you're absolutely sure you want to get rid of it, please get")
    print("rid of it explicitly (e.g. by 'rm -rf "+modeldir+"') first.")
    exit()
  # end if model already exists

  
  msgsfn=args["msgsfn"]
  if msgsfn is None:
    print("Required arg 'msgsfn' (training messages file name) is missing.")
    exit()
  if os.path.isfile(msgsfn):
    print("Training messages are in "+msgsfn+".")
  else:
    print("Training messages not found at "+msgsfn+".")
    exit()

  label=args["label"]
  if label is None:
    label="messages"
  else:
    print("Henceforth I will refer to each of these messages as "+label+".")

  description=args["description"]
  if description is None:
    description="No description provided."
  else:
    deeptext.wrapsody("... ","The short description of these "+label+":  "+description)

  minmsglen=args["minmsglen"]
  if minmsglen==0:
    minmsglen=1
  else:
    print("We'll discard "+label+" shorter than "+str(minmsglen)+" characters.")

  maxmsglen=args["maxmsglen"]
  if maxmsglen==0:
    maxmsglen=1000000
  else:
    print("We'll discard "+label+" longer than "+str(maxmsglen)+" characters.")

  if minmsglen<1 or maxmsglen>1000000 or maxmsglen<minmsglen:
    print("That's pretty silly if you think about it...")
    exit()

  msgslen,msgs=readmsgs(msgsfn,minmsglen,maxmsglen)
  print(("{:,d} "+label+" ({:,d} characters).").format(len(msgs),msgslen))
 
  # TODO I need to check these and provide defaults etc.
  hidden=args["hidden"]
  print("There are "+str(hidden)+" hidden units each of the 2 LSTM layers.")

  seqlen=args["seqlen"]
  print("The model will predict the next character based on the preceeding "+str(seqlen)+".")

  # build the model: 2 LSTM layers
  print()
  print("Creating new "+label+" model at "+modeldir+".")
  alienbrain=deeptext.cortex(modeldir,hidden=hidden,seqlen=seqlen,
      label=label,description=description,msgs=msgs)
  alienbrain.train()

# end def newmodel


def readmsgs(msgsfn,minmsglen,maxmsglen):
  """Read training messages, cleaning and filtering"""

  print()
  print("Reading "+msgsfn+"...")
  with open(msgsfn,"r") as f:
    text=f.read()
    f.close()
  rawlen=len(text)
  if rawlen==0:
    print(msgsfn+" is empty.")
    exit()

  text=unidecode.unidecode(text)
  rawmsgs=text.split("\n")
  text=None

  # Curate the messages somewhat.
  #  - Filter based on length.
  #  - Get rid of excess white space, leading/trailing spaces
  #  - Break author information into a separate array.
  msgs=[]
  msgslen=0
  for rawmsg in rawmsgs:
    msg=rawmsg.strip() 
    while "  " in msg:
      msg=msg.replace("  "," ")
    msg=msg.strip() 
    if len(msg)>=minmsglen and len(msg)<=maxmsglen:
      msgs.append(msg)
      msgslen+=len(msg)
  # for each raw message
  rawmsgs=None

  return msgslen,msgs

# end def readmsgs



if __name__ == '__main__':
  signal.signal(signal.SIGINT,deeptext.goquietly)
  newmodel()


