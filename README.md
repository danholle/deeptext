
# ***deeptext***

**Deep Learning Toolkit for Domain-Specific Text Generation**

-------

Here you will find a set of tools for creating, manipulating, analyzing, testing, and using 
deep learning text models trained from *message collections* for a domain of your choosing.  By 
"message collection", I mean collections of things like
 * tweets
 * document titles
 * headlines
 * one-liners (jokes)
 * quotations

In all these examples, the message is relatively short; is meaningful in isolation; and they conform to some sort of 
syntactic and semantic pattern which we'd like to automatically learn, so we could generate plausible new messages 
which fit these patterns.

This is different from the char-rnn family of things out there, where bodies of text are often collection of 
heterogeneous parts mashed into one giant endless flowing stream.  *deeptext*, instead, is about homogeneous collections
of tweets, titles, etc. which are modelled as separate instances rather than parts of an endless flow.

If you can collect at least few hundred KB of your own message instances into a file, one message per line, you can create models of
your messages from the command line, and test your model to see if it "gets it".  You can subsequently use Python
services to hallucinate new instances in various ways.

I'm using *deeptext* to build a deep POTUS (given a hypothetical scenario where the President's most visible manifestation
is his twitter stream);  and a deep blogger (re-imagining an existing blog, generating a new one as output).

If you're doing something where re-imagining text would be useful, and this stuff would help, that's great.

-------

# Example

Let's make this more concrete with an example.  As it happens, *deeptext* comes with 3 models I use for testing.
You can use them with *deeptext* right out of the box to try out the tools on some pre-trained models.

One of the 3 models is a collection of over 100,000 famous quotations
collected from a number of sources.  These were placed in a *message collection* with one quotation per line, in the
form "Author Name : Wise Words".

I used the *newmodel* command (with subsequent *study* commands) to create a model in models/quotes.  To see how the 
model was doing, I used the *improv* command to generate a few random quotes.  Here's what it came up with:

      Improvising some random quotations:

       -- Mary Margare : It's the same of a little relation in the man who is still       
          the soul to be a greatest who they are dead that the best thing is the          
          thing in a problem.                                                             
       -- John Laur : The art of a substance of a man is the chance to make the as it     
          will not see it.                                                                
       -- Carl Gord : I have to stay that the book of movies with a man is a way to       
          accept to be an end.                                                            
       -- Peter Lick : We don't learn a sace that I think that anything should be the     
          happy and a changes and sex and the and part of an institution of the           
          individual problems and different is the greatest history of the designer.      
       -- Alice Herbert : I was a way to say I have to be a thing to work when they       
          don't have to compared the ability to do them.                                  

The alien brain has faithfully grokked and replicated the quotation structure, complete with credible looking
names with correct capitalization.  Notice that all 5 author names were hallucinated based on the names in the test data:  none of these
names appear in the quotations file.  Even more amazing is that the quotations generated
appear to be at least somewhat coherent:  the alien brain learned EVERYTHING (spelling, punctuation, capitalization,
grammar, etc.) all just by looking at a load of character strings.

Pretty bizarre, I'd say.

-------

There's a lot more stuff I need to write here, including
 - Documenting how to set up your own *deeptext* instance
 - Documenting commands
 - Documenting example models
 - Giving coding examples

But I thought it would be best to get this out there, then add these parts as time permits.

On that note, one more quotation:

       -- Arnold Schwarzenegger : I'll be back!

-------

## Approach Used by *deeptext*

*deeptext* uses bash, Python 3, and Keras.  I've only tested with TensorFlow so far, but
I'm hoping to get this all running with CNTK, given that its LSTM training performance is
apparently so much better.

I'm developing it on Ubuntu 16.04, and have every reason to believe that it would work
fine on whatever Linux or on a Mac.  If you're on Windows, I'm afraid all I can offer
you at the moment are my condolences.

The *newmodel* script is used to create a new model.  Details are provided below;  but
in summary at this point you provide a data file (one "message" per line) and some short
descriptive text, as well as some simple network topology, and training will begin.

"Simple network topology" means sequence length (n) & number of LSTM units per layer (m).  I'll construct
a network with two LSTM layers, each with m units.  The network will take n input characters and produce
a multinomial probability distribution for the next character.  Each LSTM layer has a dropout ratio of 
0.3 because that's what the cool kids do. 

Although *newmodel* starts the training process, training is usually so long that you might need to
stop it and restart it some number of times.  Once *newmodel* has created an initial model, feel free
to croak it;  you can always resume training using the *study* command.

Various tools, such as *improv*, will use the model to do useful things, like generating random messages based
on the model (as in the example above);  or generating random completions for a message you start;  or providing
alternate completions for messages found in the training set.  Depending on your recreational drug preference,
each of these tasks is potentially entertaining.

Models are saved as 3 files in their own directory... normally a subdirectory of the *models* directory.  Your
model directory is named when newmodel is invoked.  The 3 model files are
 - A txt file containing your training data.  This file is created by newmodel and is used (among other things)
   for continuing training later.  Yeah, I know it is probably redundant, but it's proven very handy to keep this
   here so you don't have to worry if you're using the right version of the data later... and so forth.
 - A json file containing model properties, such as various titles and descriptions, topology, etc.
 - An hdf5 file containing model weights.

The training process scopes out your training data, then starts out the process with a relatively small
number of samples (like 20,000) so you can quickly get a model that starts to work.  If there's any sign
of overtraining, it will double the number of samples for the next epoch, and continue with that until
there's overtraining again... then double again and so forth.  I've found this approach to training to
be handy for training on wimpy machines without GPU's.

I think that's about all I need to say about the approach, and I hope you agree, although to be frank, 
at this stage your vote doesn't count.

-------

## Installing *deeptext*

TODO

-------

## Directory Structure

TODO

-------

## Command Line Tools

TODO

-------

## Example Models

TODO

-------

## Coding Examples

TODO


