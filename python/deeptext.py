#!/usr/bin/python3

"""Create LSTM model from a message collection."""

# TODO sample:  we should not ever get here!

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
      self.vloss=100.0
      self.tloss=100.0

      self.savemsgs()
      self.saveproperties()
    elif (hidden is None and seqlen is None and vocab is None
    and interleave is None and label is None and description is
    None and msgs is None):
      # Reading existing model from modeldir 
      self.loadproperties()
      self.loadmsgs()
    else:
      print("cortex:  Invalid constructor.  Specify all or none.")
      exit()
    # end if / breaking out on constructor type
    
    self.model=Sequential()
    self.model.add(LSTM(self.hidden, return_sequences=True,
        input_shape=(self.seqlen,self.vocabsize)))
    self.model.add(Dropout(0.3))
    self.model.add(LSTM(self.hidden))
    self.model.add(Dropout(0.3))
    self.model.add(Dense(self.vocabsize,activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam')
    #self.model.add(Dense(self.vocabsize))
    #self.model.add(Activation("softmax"))
    #optimizer=RMSprop(lr=0.01)
    #self.model.compile(loss="categorical_crossentropy",optimizer=optimizer)


    # If there are weights out there, load them.
    self.loadweights()

  # end def __init__


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
    fails=0
    while True:
      print('-' * 50)
      print("Epoch {}:  2xLSTM({}), seqlen {}, train {:,d}, interleave {}"
          .format(1+self.epochs,self.hidden,self.seqlen,self.seqcnt,self.interleave))

      print("Vectorizing...",end="")
      print("\r",end="")
      tcnt=(self.seqcnt+self.interleave-1)//self.interleave
      random.shuffle(self.deck)
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
      hist=self.model.fit(X, y, batch_size=100, validation_split=0.2,
          epochs=1)
      eloss=self.estloss()
      print("Independent model loss assessment:",eloss)
      self.epochs+=1
      self.samples+=tcnt

      # Look at training and validation loss, past and present.
      # Use these to decide things like:
      #  -  Should I save the model?
      #  -  Do I need finer-grained training data?
      # There are some mysteries here.  For example, training loss is often
      # larger than validation loss.  Regularly.  Seems strange. 
      # 
      # When do go finer-grained?
      #  - If two soft fails in a row, or one hard fail.
      # When do we save the model?
      #  - No hard fail, and
      #  - Validation improved (or, if virgin, no soft fail)
      vloss=hist.history['val_loss'][0]
      tloss=hist.history['loss'][0]
      failtype=0
      if self.vloss:
        if self.vloss<=vloss:
          print("Looks like we're overfitting.")
          failtype=2
        elif self.tloss>tloss:
          print("We're making progress:  both training and validation set improved.")
          ratio=(self.tloss-tloss)/(self.vloss-vloss)
          if ratio>3.0 and tloss<vloss:
            print("But training improved WAY more.  Overfitting.")
            failtype=2
          elif ratio>2.0:
            print("Training improved a lot more.  Hmmm...")
            failtype=1
        else: # validation improved but training did not?!
          print("Training process is wandering.")
          failtype=1
      else:
        if vloss>tloss:
          print("Training better than validation.  Hmmm...")
          failtype=1
      # Decide about saving model
      fails+=failtype
      if fails>=2:
        if self.interleave==1:
          print("...Stopping.  Time to train with more data.")
          exit()
        else:
          print("...Reverting back to saved model.")
          self.loadproperties()
          self.loadweights()
          self.vloss+=100.0
          self.tloss+=100.0 
          self.interleave=self.interleave//2
          self.saveproperties()
          print("...Reducing interleave to "+str(self.interleave)+".")
          fails=0
      else:
        if failtype==0 or self.vloss>vloss:
          self.vloss=vloss
          self.tloss=tloss
          print("Saving model...")
          self.saveproperties()
          self.saveweights()
          
          print("Let's ask the model for some random "+self.label+":")
          wrapsody(" -- ",self.improv())
          wrapsody(" -- ",self.improv())

          if failtype==0:
            fails=0
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
      self.vloss=props["vloss"]
      self.tloss=props["tloss"]
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
    props["samples"]=self.samples
    props["vloss"]=self.vloss
    props["tloss"]=self.tloss
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


  def estloss(self,sampsize=2500):
    """Make a serious attempt to check the model loss"""
    print("Estimating loss across",sampsize,"samples...",end="")
    print("\r",end="")
    
    imod=self.seqcnt//sampsize
    if imod<3:
      imod=1
    random.shuffle(self.deck)
    sumnl=0.0
    sum1=0.0

    i=0
    for inseq,outch in self.gensamples():
      i+=1
      if i%imod==0:
        if sum1%5000==0:
          print("Looking at sample",(sum1+1),"of",sampsize,"...",end="")
          print("\r",end="")
        # compute p, the probability of outch given inseq
        x=np.zeros((1,self.seqlen,self.vocabsize))
        for i,ch in enumerate(inseq):
          x[0, i, self.ch2ix[ch]] = 1
        ps=self.model.predict(x, verbose=0)[0]
        p=ps[self.ch2ix[outch]]

        if p<=0.0:
          p=0.0000001
        sumnl-=math.log(p)
        sum1+=1
      # if this sample is in our test set of size sampsize
    return sumnl/sum1
  # end def estloss


  def rewrite(self,origmsg):
    """Rewrite a message given a model.
    Returns:
      newmsg: The entire new message
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


# Print a line, the full width of the terminal (filling or trunc'ing if necessary)
def oneliner(m):
  cols,rows=os.get_terminal_size(0)
  if len(m)>cols-5:
    print(m[:cols-5]+" ... ")
  else:
    print(m.ljust(cols))
# end def oneliner


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


def showval(prefix,val):
  ''' Display a value '''
  print((prefix+":").ljust(24)+" "+val)


if __name__ == '__main__':
  intuit()


