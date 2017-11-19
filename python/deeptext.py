#!/usr/bin/python3

"""Create LSTM model from a message collection."""

# TODO remove vsamp stuff from epochresult.
# TODO clean up naming:  memfull/util, vsamps, etc

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout
from keras.optimizers import RMSprop
import json
import numpy as np
import random
import sys
import os
import argparse
import time
import psutil
import gc
import math
from unidecode import unidecode
from collections import namedtuple


class cortex(object):
  """LSTM network for grokking a message collection.
  
  Attributes:
    ch2ix (dict):  Maps chars to char# in vocab
    deck (
    description (str):  One line description of the message data
      we're using.
    epochhistory (dict):  Maps epoch# into an epochresult which
      describes what happened in that epoch.  Failed epochs
      (where optimization got lost) are included.
    hidden (int):  Number of hidden units in each LSTM layer
    interleave (int): Number of chars skipped between samples
    ix2ch (dict): maps vocab char# to actual char 
    label (str):  One word describing what we're modeling, e.g.
      jokes, tweets, quotes, headlines, ...
    modeldir (str):  Directory where the model is saved.  In this 
      directory you'll find files like
        epochhistory.json
        msgs.txt
        properties.json
        weights.hdf5
    msgs (list):  Messages in training data.  Trimmed, no 
      newlines
    seqcnt (int):  Total chars in training data.  We include
      one virtual newline in addition to string in msgs
    seqlen (int):  Number of chars in the input string to the
      function we are learning.  Moving window, if you like.    
    tsampcnt:  Number of training samples in this epoch
    valavg (real): average(-log(p)) across all vsampcnt samples
      samples in valsamps
    valsamps (list):  Validation set.  Each list item is a 
      string seqlen+1 chars long;
    vocab (str):  all the distinct chars found in msgs
    vocabsize (int):  Number of distinct characters
    vsampcnt (int):  Number of samples in the validation set


  """

  def __init__(self,modeldir,hidden=None,seqlen=None,
      label=None,description=None,msgs=None):
    """cortex constructor.
    Args:
      modeldir: directory where model resides/is built 
      hidden: # of hidden units in each LSTM layer
      seqlen: length of sequence the model uses for prediction
      vocab: All chars in the vocabulary.
      interleave: # chars to skip forward for next sample
      label: short name for these messages, e.g. quotes, jokes, etc.
      description:  a one line description of this content
      msgs: a list of messages for training.
    
    There are two basic shapes to this constructor.
     1.  Just read in the existing model from the modeldir.
         In this case, none of the other things can be specified.
     2.  Create a new model at the specified place.  In this
         case, ALL of the other things must be specified.

    """

    self.modeldir=modeldir
    if os.path.isfile(self.modeldir):
      print("cortex:  "+modeldir+" is a file, not a model directory.")
      exit()
    if not os.path.exists(self.modeldir):
      print("cortex:  Creating model directory "+modeldir+"...")
      os.makedirs(modeldir)

    # epochhistory maps epoch number into an epochresult for that epoch.
    # Epoch #0 is initialization (before optimization begins)
    # epochresult contains various stats about the epoch, including
    #   loss before and after, elapsed time, etc.  (see below)
    self.epochhistory=dict()
    self.epochresult=namedtuple("epochresult",[
        "loss0",   # Validation loss at start using vsamp0 samples 
        "loss1",   # Validation loss at end using vsamp0 samples
        "vsamp0",  # Number of samples in validation loss calculations
        "vsamp1",  # Recommened validation samples for next epoch
        "intlv0",  # Interleave for this epoch
        "intlv1",  # Recommended interleave for next epoch
        "tsamp",   # Number of samples used in training for this epoch
        "memutil", # Memory utilization of training samples 0..1.0
        "elapsed", # Elapsed time for epoch in seconds
        "saved"    # True if weights saved
      ])
    
    if (hidden is not None and seqlen is not None and label is not None
    and description is not None and msgs is not None):
      # Creating new model

      self.hidden=hidden
      self.seqlen=seqlen
      self.label=label
      self.description=description

      propsfn=os.path.join(self.modeldir,"properties.json")
      props=dict()
      props["hidden"]=self.hidden
      props["seqlen"]=self.seqlen
      props["label"]=self.label
      props["description"]=self.description
      with open(propsfn,"w") as f:
        json.dump(props,f,indent=2,sort_keys=True)
        f.close()

      self.vocab=sorted(list(set("\n".join(msgs))))
      self.vocabsize=len(self.vocab)
      self.ch2ix=dict((ch,ix) for ix,ch in enumerate(self.vocab))
      self.ix2ch=dict((ix,ch) for ix,ch in enumerate(self.vocab))

      vocfn=os.path.join(self.modeldir,"vocabulary.json")
      with open(vocfn,"w") as f:
        json.dump(self.ch2ix,f,indent=2,sort_keys=True)
        f.close()

      self.msgs=msgs
      self.deck=[i for i in range(len(msgs))]
      random.shuffle(self.deck)
      self.seqcnt=0
      for msg in self.msgs:
        self.seqcnt+=1+len(msg)

      # Set interleave so that the initial training set
      # has around 20K samples
      self.interleave=1
      self.tsampcnt=self.seqcnt
      if self.tsampcnt>30000:
        self.interleave=(self.seqcnt+10000)//20000
        self.tsampcnt=(self.seqcnt+self.interleave-1)//self.interleave

      self.valavg=math.log(self.vocabsize)
      self.remodel()
      self.savemsgs()
      self.valsamps=[]

      # Epoch 0 describes state after initialization. 
      self.epochhistory[0]=self.epochresult(
        loss0=0.0, loss1=math.log(self.vocabsize),
        vsamp0=0, vsamp1=2000, intlv0=0, intlv1=self.interleave,
        tsamp=0, memutil=0.0, elapsed=0, saved=False)
      ehfn=os.path.join(self.modeldir,"epochhistory.json")
      with open(ehfn,"w") as f:
        json.dump(self.epochhistory,f,indent=2,sort_keys=True)
        f.close()

    elif (hidden is None and seqlen is None and label is None 
    and description is None and msgs is None):
      # Reading existing model from modeldir 
      self.loadproperties()
      self.remodel()
      self.loadweights()
      self.loadmsgs()
      self.valsamps=[]
    else:
      print("cortex:  Invalid constructor.  Specify all or none.")
      exit()
    # end if / breaking out on constructor type
    
  # end def __init__


  def remodel(self):
    """Build the model structure in Keras
        model = Sequential()
        model.add(Embedding(max_features, 128, ))

        if depth > 1:
            for i in range(depth - 1):
                model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
    """
    self.model=Sequential()
    self.model.add(LSTM(self.hidden, return_sequences=True,
        input_shape=(self.seqlen,self.vocabsize)))
    self.model.add(Dropout(0.2))
    self.model.add(LSTM(self.hidden))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(self.vocabsize,activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam')
    self.model.summary()
  # end def remodel


  def remodelSRU(self):
    """Build the model structure in Keras using titu1994 SRU"""
    self.model=Sequential()
    self.model.add(LSTM(self.hidden, return_sequences=True,
        input_shape=(self.seqlen,self.vocabsize)))
    self.model.add(Dropout(0.2))
    self.model.add(LSTM(self.hidden))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(self.vocabsize,activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam')
    self.model.summary()
  # end def remodelSRU


  def gensamples(self):
    """generator function for all seq-> char mappings in train data"""
    for msgno in self.deck:
      msg=self.msgs[msgno]
      sequence=" "*self.seqlen+msg+"\n"
      while len(sequence)>self.seqlen:
        yield sequence[:self.seqlen],sequence[self.seqlen]
        sequence=sequence[1:]
  # end def gensamples


  def train(self):
    """Train the model."""    
    # TODO clean out fails & friends

    while True:
      ehlast=self.epochhistory[len(self.epochhistory)-1] # last epoch's result
      self.interleave=ehlast.intlv1
      self.vsampcnt=ehlast.vsamp1
      self.tsampcnt=(self.seqcnt+self.interleave-1)//self.interleave

      # To monitor progress in the model, the validate function 
      # squirrels away a significant (but not comprehensive) 
      # collection of samples to estimate model loss.  We use these
      # same samples from iteration to iteration.  This appears to
      # be a far more stable way of discerning incremental progress
      # than to try to build an estimator that is extremely accurate.
      #
      # First time through, valsamps is empty.  The code which follows
      # calculates the validation loss, which causes valsamps to be
      # set up with the desired number of validation samples, which 
      # reuse (per the above) as a stable loss measure.
      b4cnt=len(self.valsamps) # samples we already have in hand
      if b4cnt!=self.vsampcnt or not ehlast.saved:
        self.validate()
        tsprint(" ")
        tsprint("Captured {:,d} samples for validation.  Resulting loss is {:0.4f}."
          .format(self.vsampcnt,self.valavg))

      tshl()

      tsprint(
          "Epoch {}:  Training {} model, 2xLSTM({}), seqlen {}, starting loss={:0.4f}."
          .format(len(self.epochhistory),self.label,self.hidden,self.seqlen,self.valavg))
      tsprint(
          "Using interleave {} ({:,d} training samples)."
          .format(self.interleave,self.tsampcnt))

      flashline("Creating one-hot encoding ({:,d}x{}x{} tensor)..."
          .format(self.tsampcnt,self.seqlen,self.vocabsize))

      random.shuffle(self.deck)

      # One-Hot Encoding is memory intensive.  Monitor size we
      # need for this epoch, and compute minimum interleave
      # we have space for.

      # Check memory before
      gc.collect()
      time.sleep(1.0) # is gc async?
      gc.collect()
      time.sleep(1.0)
      memb4=psutil.virtual_memory().free/1048576.0

      X=np.zeros((self.tsampcnt,self.seqlen,self.vocabsize),dtype=np.bool)
      y=np.zeros((self.tsampcnt,self.vocabsize),dtype=np.bool)
      seqno=0
      tno=0
      for inseq,outch in self.gensamples():
        if seqno%self.interleave==0:
          for t,ch in enumerate(inseq):
            X[tno,t,self.ch2ix[ch]]=1
          y[tno,self.ch2ix[outch]]=1
          tno+=1
        seqno+=1

      # Check memory after
      gc.collect()
      time.sleep(1.0) # is gc async?
      gc.collect()
      time.sleep(1.0)
      memaft=psutil.virtual_memory().free/1048576.0
      memdelta=memb4-memaft
      if memdelta<1:
        memdelta=1
      memfull=memdelta/memb4
      maxsamp=int(self.tsampcnt*0.9/memfull)
      tsprint(
          "One-hot encoding:  {:,d}x{}x{} tensor took {:,d}MB ({:0.1f}% of memory)."
          .format(self.tsampcnt,self.seqlen,self.vocabsize,int(memdelta),
          int(100.0*memfull)))
      self.minterleave=int((self.seqcnt+maxsamp-1)/maxsamp)
      if self.minterleave<1:
        self.minterleave=1
      #tsprint("Max samples given our memory size:  {:,d} (interleave {})"
      #    .format(maxsamp,self.minterleave))
      if self.minterleave<2:
        tsprint("Memory is not a constraint for this model & data.")
      elif self.minterleave<0.35*self.interleave:
        tsprint("Memory is not a constraint, at least for a while yet.")
      else:
        tsprint("Memory may soon be a limiting factor in training this model.")

      ptprefit=time.process_time()
      pcprefit=time.perf_counter()
      hist=self.model.fit(X, y, batch_size=128, epochs=1)
      ptpostfit=time.process_time()
      pcpostfit=time.perf_counter()
      
      # Free up memory NOW so it's there for next iteration
      X=None 
      y=None

      intlvcurr=self.interleave
      felapsed=pcpostfit-pcprefit
      process=ptpostfit-ptprefit
      tsprint(
          "Training took {:0.1f} elapsed seconds using {:.1f} CPU threads."
          .format(felapsed,process/felapsed))

      # Look at the new model loss.
      preavg=self.valavg
      precnt=self.vsampcnt
      
      pcpreval=time.perf_counter()
      currcnt,curravg=self.validate()
      pcpostval=time.perf_counter()
      velapsed=pcpostval-pcpreval
      tsprint(
          "New model loss {:0.4f}, {:,d} samples, {:0.1f} seconds)."
          .format(curravg,currcnt,velapsed))

      improve=preavg-curravg
      tsprint("Improvement {:0.4f} loss units ({:0.4f} units/hour)."
          .format(improve,improve*3600.0/felapsed))
     
      # Now decide how to proceed based on the new model loss.
      # Take an optimistic view of what we are doing next
      saving=True     # Saving the model
      zoomin=False    # reducing the interleave
      reason="none"
      
      # We are saving if model improved.
      # If not, we are reverting.  Note that the optimization sometimes
      # gets lost, and ends up with a worse model than before;  this 
      # seems to be a matter of luck, and reverting / rerunning fixes it.
      m1=len(self.epochhistory)-1
      m2=len(self.epochhistory)-2
      if improve<=0.0:
        saving=False
        tsprint("Reverting back to last good model.")
        self.loadweights()
        if not self.epochhistory[m1].saved: # 2 in a row = more samps
          zoomin=True
          reason="we had two consecutive failed epochs" 
      else: # if saving
        saving=True
        tsprint("Saving the new, improved model...")
        self.saveweights()
          
        tsprint("Let's ask the model for some random "+self.label+":")
        wrapsody(" -- ",self.improv())
        wrapsody(" -- ",self.improv())

        # We start with 2 epochs with a small (~20K) sample set, just
        # to quickly flush out any problems with the job at hand.
        # 
        # From there, we "zoom in" (cut the interleave in half, doubling
        # the number of samples) under two circumstances:
        #  1.  This epoch took < 300 secs.
        #  2.  The last 3 iterations each showed a declining improvement
        #      in the loss.
        if len(self.epochhistory)>=2 and memfull<0.80:
          if felapsed<600.0:
            zoomin=True
            reason="epoch is < 10 minutes"
          else:
            eh1=self.epochhistory[m1]
            improve1=eh1.loss0-eh1.loss1
            eh2=self.epochhistory[m2]
            improve2=eh2.loss0-eh2.loss1
            if ((intlvcurr==eh1.intlv0) and (intlvcurr==eh2.intlv0)
            and (improve2>improve1) and (improve1>improve)):
              zoomin=True
              reason="we had 3 declining epochs in a row"
            # end if declining 
          # end if
        # end if we have done 3 or more epochs
      # end if saving


      # Deal with halving the interleave
      newintlv=self.interleave
      if zoomin:
        newintlv=newintlv//2
        if newintlv<self.minterleave:
          newintlv=self.minterleave
        if newintlv>=self.interleave:
          tsprint("Looks like we aren't learning anything new here.")
          exit()
        else:
          tsprint("Decreasing the interleave from {} to {}, because {}."
              .format(self.interleave,newintlv,reason))
          self.interleave=newintlv
        # end if
      # if halving


      # Compute validation sample count for next iteration
      tsampcntnew=(self.seqcnt+self.interleave-1)//self.interleave
      vsampnew=2000
      while tsampcntnew>20*vsampnew:
        vsampnew+=2000
      
      self.epochhistory[len(self.epochhistory)]=self.epochresult(
        loss0=preavg, loss1=curravg, vsamp0=currcnt, vsamp1=vsampnew, 
        intlv0=intlvcurr, intlv1=self.interleave, tsamp=self.tsampcnt, 
        memutil=memfull, elapsed=felapsed, saved=saving)
      ehfn=os.path.join(self.modeldir,"epochhistory.json")
      with open(ehfn,"w") as f:
        json.dump(self.epochhistory,f,indent=2,sort_keys=True)
        f.close()

      # Display recent progress here
      titleline="Recent progress summary:"
      epochline="  Epoch       "
      intlvline="  Interleave  "
      tsampline="  # Samples   "
      memline  ="  % Memory    "
      timeline ="  Seconds     "
      elossline="  Loss        "
      progline ="  Improvement "
      uphline  ="  Improv/Hour "
    
      m=len(self.epochhistory)-1
      n=0
      while m>0 and n<5:
        eh=self.epochhistory[m]
        epochline+="{:12d}".format(m)
        intlvline+="{:12d}".format(eh.intlv0)
        elossline+="{:12.4f}".format(eh.loss1)
        progline+="{:12.4f}".format(eh.loss0-eh.loss1)
        uphline+="{:12.4f}".format((eh.loss0-eh.loss1)*3600.0/eh.elapsed)
        timeline+="{:12.1f}".format(eh.elapsed)
        tsampline+="{:12,d}".format(eh.tsamp) 
        memline+="{:11.1f}".format(eh.memutil*100.0)+"%" 
        m-=1
        n+=1
      # end while
     
      tsprint(" ")
      tsprint(titleline)
      tsprint(epochline)
      tsprint(intlvline)
      tsprint(tsampline)
      tsprint(memline)
      tsprint(timeline)
      tsprint(elossline)
      tsprint(progline)
      tsprint(uphline)
  
    # end while forever 
  # end train
  

  def load(self):
    """Load model (properties, weights, messages) from modeldir"""
    self.loadproperties()
    self.loadweights()
    self.loadmsgs()
  # end def load


  def loadproperties(self):
 
    propsfn=os.path.join(self.modeldir,"properties.json")
    if os.path.isfile(propsfn):
      with open(propsfn,"r") as f:
        props=json.load(f)
        f.close()
      self.hidden=props["hidden"]
      self.seqlen=props["seqlen"]
      self.label=props["label"]
      self.description=props["description"]
    # end if there is a properties file in the model

    vocfn=os.path.join(self.modeldir,"vocabulary.json")
    if os.path.isfile(vocfn):
      with open(vocfn,"r") as f:
        self.ch2ix=json.load(f)
        f.close()
      self.vocabsize=len(self.ch2ix)
      self.ix2ch=dict((self.ch2ix[ch],ch) for ch in self.ch2ix)
    # end if there is a vocabulary file in the model

    ehfn=os.path.join(self.modeldir,"epochhistory.json")
    if os.path.isfile(ehfn):
      with open(ehfn,"r") as f:
        eh=json.load(f)
        f.close()
      
      # json serialization is a bit goofy.
      # keys MUST be strings (messing up our integer keys), and
      # namedtuples turn into lists.  We fix it.
      self.epochhistory=dict()
      for key in eh:
        self.epochhistory[int(key)]=self.epochresult(*eh[key])
    # end if there is an epoch history file in the model
    
  # end def loadproperties


  def loadweights(self):
    weightsfn=os.path.join(self.modeldir,"weights.hdf5")
    if os.path.isfile(weightsfn):
      self.model.load_weights(weightsfn)
  # end def loadweights


  def loadmsgs(self):
    msgsfn=os.path.join(self.modeldir,"msgs.txt")
    if os.path.isfile(msgsfn):
      with open(msgsfn,"r") as f:
        recs=f.readlines()
        f.close()
      msgs=[]
      for rec in recs:
        rec=rec.replace("\n"," ").replace("  "," ").strip()
        msgs.append(rec)
      self.msgs=msgs
      self.deck=[i for i in range(len(msgs))]
      random.shuffle(self.deck)
      self.seqcnt=0
      for msg in self.msgs:
        self.seqcnt+=1+len(msg)
    else:
      self.msgs=None
  # end def loadmsgs


  def saveweights(self):
    weightsfn=os.path.join(self.modeldir,"weights.hdf5")
    self.model.save_weights(weightsfn)
  # end def saveweights


  def savemsgs(self):
    # We save messages into model directory so we
    # can continue training at any time
    if self.msgs:
      msgsfn=os.path.join(self.modeldir,"msgs.txt")
      with open(msgsfn,"w") as f:
        for msg in self.msgs:
          f.write(msg+"\n")
        f.close()
  # end def savemsgs


  def finishmsg(self,prefix,maxlen=240,temperature=0.4,
      pratio=0.1):
    """Complete a message given text it starts with.
    Args:
      prefix:  a partial message
      maxlen:  longest message you'd like to get back.
    Returns:
      newmsg: The entire new message
    """

    msg=prefix

    done=False
    while not done:
      seq=(" "*self.seqlen+msg)[-self.seqlen:]
      x=np.zeros((1,self.seqlen,self.vocabsize))
      for i,ch in enumerate(seq):
        x[0, i, self.ch2ix[ch]] = 1
      ps=self.model.predict(x, verbose=0)[0]
      ix=self.sample(ps,temperature=temperature,pratio=pratio)
      ch=self.ix2ch[ix]
      if ch=="\n":
        done=True
      else:
        msg+=ch
        if len(msg)>=maxlen:
          msg=msg[:maxlen-5]+"  ..."
          done=True
    # end while not done

    return msg
  # end finishmsg


  def validate(self):
    """Make a serious attempt to check the model loss.

    This is done by looking at samples spaced across the training set.
    We keep these samples so subsequent estloss calls with the same
    sample count use the same samples... so we have a stable measure
    i.e. if the model doesn't change we get the same loss estimate.

    Returns: 
      Average perplexity (minus log p) for that set of samples.


    """
    flashline("Estimating loss across {:,d} samples...".format(self.vsampcnt))

    # Remember sample size and set of samples used to compute loss.
    # If not there, or not the same # samples as last time, compute
    # a new set of samples with the correct size.
    if len(self.valsamps)!=self.vsampcnt:
      self.valsamps=[]

      # Choose a skip step that gives us twice as many samples as
      # we actually need.  This is because we make sure every sample
      # is unique so we want a generous over-supply
      imod=self.seqcnt//(2*self.vsampcnt)
      if imod<3:
        imod=1
      random.shuffle(self.deck)

      i=0
      for inseq,outch in self.gensamples():
        i+=1
        if i%imod==0 and len(self.valsamps)<self.vsampcnt:
          cand=inseq+outch
          if cand not in self.valsamps:
            self.valsamps.append(cand)
          # end if this candidate is not already in sample set
        # end if this might be a good sample to add
      # end for all samples in the training set
    # end if we don't already have a sample set for estloss
    
    # moments of neg log
    sum0=0.0
    sum1=0.0

    # Do predict in one huge go
    x=np.zeros((self.vsampcnt,self.seqlen,self.vocabsize))
    y=[]
    for i in range(self.vsampcnt):
      outch=self.valsamps[i][-1:]
      inseq=self.valsamps[i][:self.seqlen]
      for j,ch in enumerate(inseq):
        x[i,j,self.ch2ix[ch]] = 1
      y.append(self.ch2ix[outch])
    ps=self.model.predict(x,verbose=0)
    for i in range(self.vsampcnt):
      p=ps[i][y[i]]

      if p<=0.0:
        p=0.0000001
      nl=-math.log(p)
      sum0+=1.0
      sum1+=nl
    # for all samples in the test set

    # Now turn sum0 / sum1 / sum2 into stats which are more useful
    self.valavg=sum1/sum0
   
    return self.vsampcnt,self.valavg
  # end def validate


  def rewrite(self,origmsg):
    """Rewrite a message given a model.
    Args:
      origmsg:  Message we are rewriting
    Returns:
      newmsg: The new message
    """

    words=origmsg.split(" ")
    if len(words)<2:
      return origmsg
  
    prefix=" ".join(words[:(len(words)+2)//3])
    msg=prefix

    done=False
    while not done:
      seq=(" "*self.seqlen+msg)[-self.seqlen:]
      x=np.zeros((1,self.seqlen,self.vocabsize))
      for i,ch in enumerate(seq):
        x[0, i, self.ch2ix[ch]] = 1
      ps=self.model.predict(x, verbose=0)[0]
      ix=self.sample(ps)
      ch=self.ix2ch[ix]
      if ch=="\n":
        done=True
      else:
        msg+=ch
    # end while not done

    return msg
  # end rewrite


  def improv(self):
    """Improvise a message from nothing.
    Returns:
      msg:  a new message
    """

    msg=""
    done=False
    while not done:
      seq=(" "*self.seqlen+msg)[-self.seqlen:]
      x=np.zeros((1,self.seqlen,self.vocabsize))
      for i,ch in enumerate(seq):
        x[0, i, self.ch2ix[ch]] = 1
      ps=self.model.predict(x,verbose=0)[0]
      ix=self.sample(ps)
      ch=self.ix2ch[ix]
      if len(msg)>250:
        msg=msg[:240]+"..."
        ch="\n"
      if ch=="\n":
        done=True
      else:
        msg+=ch
    # end while not done

    return msg
  # end improv


  def sample(self,ps,temperature=0.4,pratio=0.1):
    """Generate a random character given probability distribution"""
    
    # Find the largest p 
    maxp=0.0
    for p in ps:
      if p>maxp:
        maxp=p
    if maxp==0.0:
      print("cortex.sample:  all probabilities 0!")
      exit()
    
    # Make pd an index to probability mapping where the
    # probabilities are all > 0 and >= maxp * maxpratio 
    pd=dict()
    sumpd=0.0
    for ix,p in enumerate(ps):
      if p>0.0 and p>=maxp*pratio:
        pd[ix]=p
        sumpd+=p
    for ix in pd:
      pd[ix]/=sumpd

    # Apply temperature to pd
    if temperature!=1.0:
      sumpd=0.0
      for ix,p in pd.items():
        newp=np.exp(np.log(p)/temperature)
        sumpd+=newp
        pd[ix]=newp
      for ix in pd:
        pd[ix]/=sumpd
    # end if temperature needs to be applied

    # Return a random value with this distribution
    z=random.random()
    for ix,p in pd.items():
      z-=p
      if z<=0.0:
        return ix
    print("sample:  we should not ever get here!")
    exit()
  # end def sample

# end class cortex


class beamer(object):
  """Contains current state of a beam search for the Perfect Message.

  Client flow will be something like this:
    Create a beamer
    Prime it with a single incomplete message (primetext)
    While crunched message collection contains incompletes
      Advance all incomplete beam messages by one char
 
  After this is done, the best message is at the top i.e. msgs[0]    
  """

  
  def __init__(self,beamwidth):
    """beamer constructor.
    Args:
      beamwidth:  width of the beam search
    """
    self.beamwidth=beamwidth
    self.count=0 # number of messages in the beam now
    self.pss=[] # p's for each char decision in msg
    self.scores=[] # score for msg, typically avg -log(p)
    self.states=[] # 0=incomplete, 1=complete message, 2=deleted
    self.msgs=[] # Message so far, including prime text, if any
  # end def __init__


  def add(self,ps,score,state,msg):
    """Add a message fragment to the mix.
    Args:
      ps:     p's for each char decision. For incomplete msg,
              len(ps)=len(msg); for complete, =len(msg)+1 
      score:  Score.  Maybe just perplexity computed from ps.  Lower=better. 
              Computed by client;  used by beamer to crunch
      state:  0=incomplete, 1=complete, 2=deleted (used) entry
      msg:    The whole message (prefix+suffix).
    """
    self.pss.append(ps)
    self.scores.append(score)
    self.states.append(state)
    self.msgs.append(msg)
  # end def add


  def crunch(self):
    """Look at all our candidates, and keep the best beamwidth of them.
    Returns:
      Count of incomplete messages in the beam
    """
    newpss=[]
    newscores=[]
    newstates=[]
    newmsgs=[]

    best=0
    incompletes=0
    while len(newmsgs)<self.beamwidth and best>=0:
      best=-1
      for j in range(len(self.msgs)):
        if self.states[j]!=2 and (best<0 or self.scores[best]>self.scores[j]):
          best=j
      if best>=0:
        newpss.append(self.pss[best])
        newscores.append(self.scores[best])
        newstates.append(self.states[best])
        newmsgs.append(self.msgs[best])
        if self.states[best]==0:
          incompletes+=1
        self.states[best]=2
      # end if we found a new best
    # while we don't have beamwidth good messages  

    self.pss=newpss
    self.scores=newscores
    self.states=newstates
    self.msgs=newmsgs
    self.count=len(self.msgs)

    return incompletes
  # end def crunch


  def pop(self):
    """Return an incomplete message which the caller will then to extend.
    Returns:
        ps   probabilities for this message
        msg  text for this message
    """
    i=0
    while i<self.count:
      if self.states[i]==0:
        self.states[i]=2
        return self.pss[i],self.msgs[i]
      i+=1
    return None,None
  # end def pop
  

  def best(self):
    """Return the winning message.
    Returns:
      score
      msg
    """
    bestmsgs=[]
    bestscores=[]
    for i in range(self.count):
      if self.states[i]==1:
        bestmsgs.append(self.msgs[i])
        bestscores.append(self.scores[i])
    return bestmsgs,bestscores
  # done def best
   
# end class beamer  


def oneliner(m):
  """Print a line.  
  Like print(m) but it fills the line, or truncates if needed, based
  on current terminal width.  This is used when there are flashline's
  around so we don't get residual chars at the end of lines.
  """  
  cols,rows=os.get_terminal_size(0)
  if len(m)>cols-5:
    print(m[:cols-5]+"  ...")
  else:
    print(m.ljust(cols))
# end def oneliner


def flashline(m):
  """Print a line without advancing.
  This is used to present current state information that might be 
  continuously updated in place as computation proceeds.  To then
  present normally, use oneliner so that any chars we have on the
  line left by flashline get overwritten.
  """
  cols,rows=os.get_terminal_size(0)
  if len(m)>cols-5:
    print(m[:cols-5]+"  ...",end="")
  else:
    print(m.ljust(cols),end="")
  print("\r",end="")
# end def flashline


# Print prefix + string, wrapping and indenting string if necessary
def wrapsody(p,s):
  indentlen=len(p)
  indent=" "*indentlen
  cols,rows=os.get_terminal_size(0)
  if s is not None:
    if cols<10+indentlen:
      oneliner("Indent "+str(indentlen)+" with "+str(cols)+" cols?  WTF?")
      exit()
    else:
      ss=p+s 
      ss=ss.replace("\n"," ")
      while ss.endswith(" "):
        ss=ss[:-1]
      while ss is not None:
        if len(ss)>cols-5:
          m=cols-5
          n=ss.rfind(" ",0,cols-4)
          if cols-5-n<0.25*(cols-5-indentlen):
            m=n 
          oneliner(ss[0:m])
          ss=ss[m:].strip()
          if ss=="":
            ss=None
          else:
            ss=indent+ss
        else:
          oneliner(ss)
          ss=None
        # end if line too long
      # end while not done
    # end if line length valid
  # end if output line exists
# end def wrapsody


def tsprint(s):
  """Print a timestamped status message, wrapping politely"""
  wrapsody(time.strftime("%X")+" ",s)
# end def tsprint


def tshl():
  """Print timestamped horizontal line cognizant of screen width"""
  cols,rows=os.get_terminal_size(0)
  now=time.strftime("%X")
  print(now)
  msg=now+" "+"-"*200
  print(msg[:cols])
  print(now)
# end def tshl


def showval(prefix,val):
  ''' Display a value '''
  print((prefix+":").ljust(24)+" "+val)


def goquietly(signal,frame):
  print()
  print()
  print("Execution interrupted.")
  print()
  sys.exit(0)


if __name__ == '__main__':
  print("What?!  I am normally import'd, not executed.")


