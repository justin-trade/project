import pandas as pd 
import matplotlib.pyplot as plt
import seaborn

import indicoio
indicoio.config.api_key = 'a9f7dedc82df0b3a2b9a9ade646ad200'

seaborn.set_style("darkgrid")

input_text = """Rafiki puts the juice and sand he collects on 			
                Simba's brow---a ceremonial crown. He then picks 
                Simba up and ascends to the point of Pride Rock. 
                Mufasa and Sarabi follow. With a crescendo in the 
                music and a restatement of the refrain, Rafiki 
                holds Simba up for the crowd to view."""

score = indicoio.sentiment([input_text])  # make an API call to get sentiment score for the input text

import re
import string

a = open('django.txt', 'r')    
text = a.read()                           # reading script

for char in string.punctuation:           # remove punctuations
    text = text.replace(char, ' ')

words = [word for word in text.split( )]  # string to a List of words

score = indicoio.sentiment([text])  # make an API call to get sentiment score for the input text

print(score)


def sample_window(seq, window_size = 10, stride = 1):  

    for pos in xrange(0, len(seq), stride):
        yield seq[pos : pos + window_size]

def merge(seq, stride = 4):

    for pos in xrange(0, len(seq), stride):
        yield seq[pos : pos + stride] 

d = {}  # dictionary to store results (regardless of story lengths)


# Parse text
delim = " "
words = [s for s in input_text.split()]  # simplest tokenization method

# Merge words into chunks
### Note: this defines the granularity of context for sentiment analysis, 
###  might be worth experimenting with other sampling strategies!
merged_words = [" ".join(w) for w in merge(words, 5)]


# Sample a sliding window of context
delim = " "
samples = [delim.join(s) for s in sample_window(merged_words, 3, 1)] 
print(samples)  # comment this line out for big input!
d['samples'] = samples



# Score sentiment using indico API
print("\n  Submitting %i samples..." % (len(samples)))
scores = indicoio.sentiment(samples)
d['scores'] = scores
print("  ...done!")


df = pd.DataFrame()
for k,v in d.iteritems():  
    df[k] = pd.Series(v)  # keys -> columns, values -> rows
df.plot(figsize = (16,8))
plt.show()
print df  # display the table of values

