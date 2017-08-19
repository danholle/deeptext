
***deeptext***
==============

*Experiments in neural message generation*

Here you will find a set of tools for creating, manipulating, analyzing, testing, and using 
deep learning text models trained from *message collections* for a domain of your choosing.  
By "message collection", I mean collections of things like
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

If you can collect a few hundred KB of your own message instances into a file, one message per line, you can create models of
your messages from the command line, and test your model to see if it "gets it".  You can subsequently use Python
services to hallucinate new instances in various ways.

I'm using this to build a deep POTUS (given a hypothetical scenario where the President's most visible manifestation
is his twitter stream);  and a deep blogger (re-imagining an existing blog, generating a new one as output).

-------

# Example

There are 3 models in this repository that I use for testing.  One is a collection of over 100,000 famous quotations
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

# Installing *deeptext*

TODO

-------

# Directory Structure

TODO

-------

# Command Line Tools

TODO

-------

# Example Models

TODO

-------

# Coding Examples

TODO


