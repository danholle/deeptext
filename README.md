
***deeptext***
==============

*Experiments in neural message generation*


-------


It is fascinating what you get when you feed collections of text into a Deep Learning algorithm.  There is this 
alien intelligence which learns in a remarkable way:  remarkable in what it *does* learn, and remarkable in
what it does *not* learn.

This is a playground for tools for creating, manipulating, analyzing, and using domain-specific training
material comprised of *message collections* for a domain of your choosing.  
By "message collection", I mean collections of things like
 * tweets
 * document titles
 * headlines
 * one-liners (jokes)
 * quotations

In all these examples, the message is relatively short; is meaningful in isolation; and they conform to some sort of 
syntactic and semantic pattern which we'd like to automatically learn, so we could generate plausible new messages 
which fit these patterns.
This is different from the char-rnn family of things out there, where bodies of text are one giant endless flowing 
amalgam.  *deeptext*, instead, is about imagining new tweets, new titles, new jokes, and new quotations based on past ones.

This project contains tools for building LSTM models of these message collections, and viewing what they're "thinking".  
It includes a few example models (headlines, quotations and jokes).  Elsewhere, I'm working on other projects for Deep President 
(in a purely fictitious world where the main thing a President does is tweet);  and Deep Blog (generating a re-imagined 
blog from an existing one).

Let the games begin.


-------


I've got a lot of writing to do... to give you more context, to help you set
up an environment where you can use this stuff, a summary of commands, etc.

But I thought it would be best to get something on github first, then embellish
this README, rather than waiting for that magic moment of perfection.

Meanwhile, however, here's a little taster. 

I mentioned that there are 3 test models (with associated training data) here.  One of them is
a large collection of famous quotations.  Each line in the training data is a "message" of the form "author : wise words", and there's
over 100,000 of them in the training set.

Using the newmodel and study tools, I created a model of these quotations over a period of about a day on my laptop.  Under the
covers, the alien brain has 2 LSTM layers of 300 units each, with a sequence length of 60 characters.

Using the improv tool, I asked the alien brain to imagine some new quotations.  Here is the output it generated:

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

On that note, one more quotation:

       -- Arnold Schwarzenegger : I'll be back!


-------


# Directory Structure

TODO


-------


# Command Line Tools

TODO


