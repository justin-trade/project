
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import sys
import os
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import indicoio
indicoio.config.api_key = '1e2290cd633a122823dd09652a766b3c'


""" Change script files to read here"""


# define your corpus here as a list of text files
corpus = [
          "one_flew_over.txt",
          "wolf_of_wallstreet.txt",
          "crazy_stupid.txt"]


# New dict to hold data
d = {}


def sample_window(seq, window_size = 10, stride = 1):  

    for pos in xrange(0, len(seq), stride):
        yield seq[pos : pos + window_size]

def merge(seq, stride = 4):

    for pos in xrange(0, len(seq), stride):
        yield seq[pos : pos + stride] 

# Map names to input files on filesystem
root_fp = os.getcwd()
corpus_fp = os.path.join(root_fp, "texts")   # put your text files in ./texts
# print("Looking for input text files: '%s'" % corpus_fp)

for t in corpus:
    fp = os.path.join(corpus_fp, t)
    print(" Reading '%s'" % t)
    with open(fp, 'rb') as f:
        text_name = t.split(".")[0]  # strip .txt file extensions
        sample_col = text_name + "_sample"
        score_col = text_name + "_sentiment"
        lines = []  # list to receive cleaned lines of text
        
        # Quick text cleaning and transformations
        for line in f:
            if str(line) == str(""): # there are many blank lines in movie scripts, ignore them
                continue
            else:
                line = line.replace("\n", " ").lower().strip().strip('*')  # chain any other text transformations here
                
                
                text2 = re.sub(r'\d+', '', line)                # STRIPPING NUMBERS
                line = text2
                
                tokens = word_tokenize(line)                    # tokenizing line
                no_stops = [t for t in tokens if t not in stopwords.words('english')]   # removing all stop words
                line = " ".join(j for j in no_stops)            
                lines.append(line)

        print("  %i lines read from '%s' with size: %5.2f kb" % (len(lines), t, sys.getsizeof(lines)/1024.))
        
        


        # Construct a big string of clean text
        text = " ".join(line for line in lines)
    
        # split on sentences (period + space)
        delim = ". "
        sentences = [_ + delim for _ in text.split(delim)]  # regexes are the more robust (but less readable) way to do this...
        merged_sentences = [delim.join(s) for s in merge(sentences, 10)]  # merge sentences into chunks
        
        # split on words (whitespace)
        delim = " "
        words = [_ for _ in text.split()]
        merged_words = [" ".join(w) for w in merge(words, 120)]  # merge words into chunks

        
        # Generate samples by sliding context window
        delim = " "
        samples = [delim.join(s) for s in sample_window(merged_words, 10, 1)]
        d[sample_col] = samples
        
        print("   submitting %i samples for '%s'" % (len(samples), text_name))
        
        # API to get scores
        scores = indicoio.emotion(samples)
        df_scores = pd.DataFrame(scores)
        d[score_col] = df_scores['joy']
        

print("\n...complete!")

df = pd.DataFrame()
# for k,v in d.iteritems():
for k,v in sorted(d.iteritems()):  # sort to ensure dataframe is defined by longest sequence, which happens to be Aladdin
    df[k] = pd.Series(v)  # keys -> columns; rows -> columns

#print(df)

""" When adding new movie add combos """

# Pick out a few stories to compare visually
combo = pd.DataFrame()
#combo['django_sentiment'] = df['django_sentiment']							# modify line according to move name
combo['one_flew_over_sentiment'] = df['one_flew_over_sentiment']
combo['wolf_of_wallstreet_sentiment'] = df['wolf_of_wallstreet_sentiment']
combo['crazy_stupid_sentiment'] = df['crazy_stupid_sentiment']
#combo['crazy_stupid_love_sentiment'] = df['crazy_stupid_love_sentiment']



ax2 = combo.plot(colormap='jet', figsize = (16,8))  # ignore mistmatched sequence length
ax2.set_xlabel("sample")
ax2.set_ylabel("joy_score")




""" Finding which movie is more generally "happy". Threshold sentiment = 0.5 """


dict2 = {}


for column in df:
    count = 0             # initializing counter
    score = 0
    
    if column.find('sentiment') != -1 :                     # iterate only over sentiment column

        for index, row in df.iterrows():
            if row[column] >= 0.5:                # threshold
                count = count + 1.0

        a = df[column].notnull()        # finding out missing values
        no_fragments = len(df[a])       # number of fragments excluding missing values

        score = count/no_fragments      # the score is percentage of number fragments that have sentiments > threshold


        dict2[column + "_score"] = score


movie_highest_score = max(dict2, key=dict2.get)
movie_lowest_score = min(dict2, key=dict2.get)


print("\n Comparions of general happiness of movies: \n")
print(movie_highest_score + " is the most generally 'happy' with a score of ", str(dict2[movie_highest_score]))
print(movie_lowest_score + " is the least generally 'happy' with a score of ", str(dict2[movie_lowest_score]))
print("all the scores: ", dict2)


plt.show()























