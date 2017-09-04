#!/usr/bin/python3

"""Create LSTM model from a message collection."""

# TODO detect and handle the occasional blowups we get
# where an epoch degrades / diverges dramatically.
# TODO cap sample count based on available RAM 

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


class cortex(object):
  """LSTM network for grokking a message collection."""

  def __init__(self,modeldir,hidden=None,seqlen=None,vocab=None,
      interleave=None,label=None,description=None,msgs=None):
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
    
    if (hidden is not None and seqlen is not None and vocab is not
    None and interleave is not None and label is not None and 
    description is not None and msgs is not None):
      # Creating new model

      self.hidden=hidden
      self.seqlen=seqlen

      self.vocab=sorted(list(set(vocab)))
      self.vocabsize=len(self.vocab)
      self.ch2ix=dict((ch,ix) for ix,ch in enumerate(self.vocab))
      self.ix2ch=dict((ix,ch) for ix,ch in enumerate(self.vocab))

      self.interleave=interleave
      self.label=label
      self.description=description

      self.msgs=msgs
      self.deck=[i for i in range(len(msgs))]
      random.shuffle(self.deck)
      self.seqcnt=0
      for msg in self.msgs:
        self.seqcnt+=1+len(msg)
      
      self.epochs=0
      self.samples=0

      self.valcnt=10000
      self.valavg=math.log(self.vocabsize)
      self.valsd=self.valavg*0.05
      self.progress="..."
      self.remodel()
      self.savemsgs()
      self.saveproperties()
      self.valsamps=[]
    elif (hidden is None and seqlen is None and vocab is None
    and interleave is None and label is None and description is
    None and msgs is None):
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
    """Build the model structure in Keras"""
    self.model=Sequential()
    self.model.add(LSTM(self.hidden, return_sequences=True,
        input_shape=(self.seqlen,self.vocabsize)))
    self.model.add(Dropout(0.2))
    self.model.add(LSTM(self.hidden))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(self.vocabsize,activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam')
  # end def remodel


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

    fails=0
    while True:
      tcnt=(self.seqcnt+self.interleave-1)//self.interleave

      # To monitor progress in the model, the validate function 
      # squirrels away a significant (but not comprehensive) 
      # collection of samples to estimate model loss.  We use these
      # same samples from iteration to iteration.  This appears to
      # be a far more stable way of discerning incremental progress
      # than to try to build an estimator that is extremely accurate.
      b4cnt=len(self.valsamps) # samples we already have in hand
      if b4cnt<0.05*tcnt and b4cnt<20000:
        self.valcnt=int(0.1*tcnt)
        if self.valcnt>20000:
          self.valcnt=20000
        if self.valcnt<2000:
          self.valcnt=2000
        self.validate()
        tsprint("Model loss {:0.3f} (sd {:0.3f}) based on {} samples."
          .format(self.valavg,self.valsd,self.valcnt))

      tsprint(" ")
      tsprint("="*50)
      tsprint(
          "Epoch {}:  Training {} model, 2xLSTM({}), seqlen {}, starting loss={:0.3f}."
          .format(1+self.epochs,self.label,self.hidden,self.seqlen,self.valavg))
      tsprint(
          "Using interleave {} ({:,d} training samples)."
          .format(self.interleave,tcnt))

      flashline("Creating one-hot encoding ({}x{}x{} tensor)..."
          .format(tcnt,self.seqlen,self.vocabsize))

      random.shuffle(self.deck)

      # One-Hot Encoding is memory intensive.  Monitor size we
      # need for this epoch, and compute minimum interleave
      # we have space for.

      # Check memory before
      gc.collect()
      time.sleep(1.0) # is gc async?
      memb4=psutil.virtual_memory().free/1048576.0

      X=np.zeros((tcnt,self.seqlen,self.vocabsize),dtype=np.bool)
      y=np.zeros((tcnt,self.vocabsize),dtype=np.bool)
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
      memaft=psutil.virtual_memory().free/1048576.0
      memdelta=memb4-memaft
      if memdelta<1:
        memdelta=1
      memfull=memdelta/memb4
      maxsamp=int(tcnt*0.9/memfull)
      tsprint(
          "One-hot encoding:  {}x{}x{} tensor took {}MB ({}% of memory)."
          .format(tcnt,self.seqlen,self.vocabsize,int(memdelta),
          int(100.0*memfull)))
      self.minterleave=int((self.seqcnt+maxsamp-1)/maxsamp)
      if self.minterleave<1:
        self.minterleave=1
      #tsprint("Max samples given our memory size:  {} (interleave {})"
      #    .format(maxsamp,self.minterleave))
      if self.minterleave<2:
        tsprint("Memory is not a constraint for this model & data.")
      elif self.minterleave<0.2*self.interleave:
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

      felapsed=pcpostfit-pcprefit
      process=ptpostfit-ptprefit
      tsprint(
          "Training took {:0.3f} elapsed seconds using {:.2f} threads."
          .format(felapsed,process/felapsed))

      # Look at the new model loss.
      preavg=self.valavg
      presd=self.valsd
      precnt=self.valcnt
      
      pcpreval=time.perf_counter()
      currcnt,curravg,currsd=self.validate()
      pcpostval=time.perf_counter()
      velapsed=pcpostval-pcpreval
      tsprint(
          "New model loss {:0.3f} (sd {:0.3f}, {} samples, {:0.3f} seconds)."
          .format(curravg,currsd,currcnt,velapsed))

      improve=preavg-curravg
      tsprint("Improvement {:0.4f} loss units ({:0.4f} units/hour)."
          .format(improve,improve*3600.0/felapsed))
     
      # Now decide how to proceed based on the new model loss.
      #  1.  If we improved by at least 0.01, we are Happy; save 
      #      everything and continue normally.
      #  2.  If we DEGRADED by at least 0.01, we are Sad;  we've
      #      presumably experienced one of those optimization blowups 
      #      that seem to happen from time to time, and revert back 
      #      to previous model and try again (changing nothing else)
      #  3.  Otherwise, we are Meh.  We don't save the new model. 
      # 
      #  If we are not Happy for twice in a row, we halve the 
      #  interleave (doubling the samples per epoch).  

      # Take an optimistic view of what we are doing next
      saving=True # Saving the model
      reverting=False # reverting to saved model
      halving=False # reducing the interleave
      
      progch=" " # are we Happy or Sad?
      if improve>0.01:
        progch="H"
      elif improve<-0.01:
        progch="S"
        saving=False
        reverting=True
      else:
        progch="M"
        saving=False
      self.progress+=progch
      if "H" not in self.progress[-2:]:
        halving=True
      if len(self.progress)>25:
        tsprint("Progress (Happy vs Sad): ..."+self.progress[-20:])
      else:
        tsprint("Progress (Happy vs Sad): "+self.progress)
 
      # Deal with halving the interleave
      newintlv=self.interleave
      if halving:
        newintlv=newintlv//2
        if newintlv<self.minterleave:
          newintlv=self.minterleave
        if newintlv>=self.interleave:
          tsprint("Looks like we aren't learning anything new here.")
          exit()
        else:
          tsprint("Decreasing the interleave from {} to {}."
              .format(self.interleave,newintlv))
          self.interleave=newintlv
      # if halving

      # Dealing with revert-to-old-model things
      if reverting:
        tsprint("Reverting back to last good model.")
        self.loadproperties()
        self.progress+=progch
        self.loadweights()
  
      # Dealing with saving model things
      if saving:  
        self.epochs+=1
        self.samples+=tcnt

        tsprint("Saving the new, improved model...")
        self.saveproperties()
        self.saveweights()
          
        tsprint("Let's ask the model for some random "+self.label+":")
        wrapsody(" -- ",self.improv())
        wrapsody(" -- ",self.improv())
      # end if saving
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
      self.interleave=props["interleave"]
      self.epochs=props["epochs"]
      self.samples=props["samples"]
      self.progress=props["progress"]
      self.valcnt=props["valcnt"]
      self.valsd=props["valsd"]
      self.valavg=props["valavg"]
      self.ch2ix=props["vocab"]
      self.label=props["label"]
      self.description=props["description"]
      self.vocabsize=len(self.ch2ix)
      self.ix2ch=dict((self.ch2ix[ch],ch) for ch in self.ch2ix)
    # end if there is a properties file in the model
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


  def save(self):
    self.saveweights()
    self.saveproperties()
    self.savemsgs()
  # end def save

  
  def saveweights(self):
    weightsfn=os.path.join(self.modeldir,"weights.hdf5")
    self.model.save_weights(weightsfn)
  # end def saveweights


  def saveproperties(self):
    """Write the properties.json file in the model directory"""

    propsfn=os.path.join(self.modeldir,"properties.json")

    props=dict()
    props["hidden"]=self.hidden
    props["seqlen"]=self.seqlen
    props["vocab"]=self.ch2ix
    props["interleave"]=self.interleave
    props["epochs"]=self.epochs
    props["progress"]=self.progress
    props["samples"]=self.samples
    props["valcnt"]=self.valcnt
    props["valsd"]=self.valsd
    props["valavg"]=self.valavg
    props["label"]=self.label
    props["description"]=self.description
    with open(propsfn,"w") as f:
      json.dump(props,f,indent=2,sort_keys=True)
      f.close()
  # end def saveproperties


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
    flashline("Estimating loss across {} samples...".format(self.valcnt))

    # Remember sample size and set of samples used to compute loss.
    # If not there, or not the same # samples as last time, compute
    # a new set of samples with the correct size.
    if len(self.valsamps)!=self.valcnt:
      self.valsamps=[]

      # Choose a skip step that gives us twice as many samples as
      # we actually need.  This is because we make sure every sample
      # is unique so we want a generous over-supply
      imod=self.seqcnt//(2*self.valcnt)
      if imod<3:
        imod=1
      random.shuffle(self.deck)

      i=0
      for inseq,outch in self.gensamples():
        i+=1
        if i%imod==0 and len(self.valsamps)<self.valcnt:
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
    sum2=0.0

    # Do predict in one huge go
    x=np.zeros((self.valcnt,self.seqlen,self.vocabsize))
    y=[]
    for i in range(self.valcnt):
      outch=self.valsamps[i][-1:]
      inseq=self.valsamps[i][:self.seqlen]
      for j,ch in enumerate(inseq):
        x[i,j,self.ch2ix[ch]] = 1
      y.append(self.ch2ix[outch])
    ps=self.model.predict(x,verbose=0)
    for i in range(self.valcnt):
      p=ps[i][y[i]]

      if p<=0.0:
        p=0.0000001
      nl=-math.log(p)
      sum0+=1.0
      sum1+=nl
      sum2+=nl*nl
    # for all samples in the test set

    # Now turn sum0 / sum1 / sum2 into stats which are more useful
    #self.valcnt=sum0
    self.valavg=sum1/sum0
    sumerr2=sum2-sum1*sum1/sum0
    if sumerr2<=0.0:
      self.valsd=0.0
    else:
      self.valsd=math.sqrt(sumerr2/(sum0*(sum0-1.0)))
   
    return self.valcnt,self.valavg,self.valsd
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


def showval(prefix,val):
  ''' Display a value '''
  print((prefix+":").ljust(24)+" "+val)


if __name__ == '__main__':
  print("What?!  I am normally import'd, not executed.")


