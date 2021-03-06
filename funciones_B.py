# librerias

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te
from nrclex import NRCLex
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import spacy
from collections import Counter

# funciones para el analisis de sentimiento y emociones

def sentiment_scores(lyrics):
    '''
    Function that calculates the sentiment
    analysis score of a song.
    '''
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(lyrics)
    #print("Overall sentiment dictionary is : ", sentiment_dict)
    
    if sentiment_dict['compound'] >= 0.05 :
        return f"{sentiment_dict['pos']*100} % Positive"
        
    elif sentiment_dict['compound'] <= - 0.05 :
        return f"{sentiment_dict['neg']*100} % Negative"
        
    else :
        return f"{sentiment_dict['neu']*100} % Neutral"       

def emotional_scores(lyrics):
    '''
    Function that calculates the emotional
    analysis scores of a song.
    '''
    df['Emotion_nrclex'] = [NRCLex(i).raw_emotion_scores for i in df['Lyrics']]
    return df 
    
# funciones para el conteo de frecuencias de palabras    
    
def plot_year_counts(df, X, y, title):
    '''
    Function to plot with a matplotlib lineplot 
    the year count summaries.
    '''
    characteristics = df.groupby(X).count()
    mpl.rcParams['figure.figsize'] = (35,10,)
    #all_songs.groupby('Year').count().plot(kind='bar')
    sns.barplot(y=characteristics[y], x=characteristics.index)
    plt.title(title)
    plt.ylabel("Number of Songs")
    plt.xticks(rotation=90)

def spacy_extraction(df, feature_column):
    '''
    Function that extracts the verb, adverb, noun, 
    and stop word Parts of Speech (POS) 
    tokens and insert them into a new dataset. 
    '''
    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = spacy.load('en_core_web_sm')
    
    for i in range (0, len(df)):
        print("Extracting information from record {} of {}".format(i+1, len(df)), end = "\r")
        song = df.iloc[i][feature_column]
        doc = nlp(song)
        spacy_dataframe = pd.DataFrame()
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "Word": token.text,
                "Lemma": lemma,
                "PoS": token.pos_,
                "Stop Word": token.is_stop
            }
            spacy_dataframe = spacy_dataframe.append(row, ignore_index = True)
        verbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "VERB"].values))
        nouns.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "NOUN"].values))
        adverbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "ADV"].values))
        corpus_clean = " ".join(spacy_dataframe["Lemma"][spacy_dataframe["Stop Word"] == False].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
    df['Verbs'] = verbs
    df['Nouns'] = nouns
    df['Adverbs'] = adverbs
    df['Corpus'] = corpus
    return df

def counting_words(df):
    '''
    Function that counts the words and 
    unique worrds of a given text.
    '''
    word_counts = []
    unique_word_counts = []
    for i in range (0, len(df)):
        word_counts.append(len(df.iloc[i]['Lyrics'].split()))
        unique_word_counts.append(len(set(df.iloc[i]['Lyrics'].split())))
    df['Word Counts'] = word_counts
    df['Unique Word Counts'] = unique_word_counts
    return df

def summary_df(df):
    '''
    Function that makes a summary of the 
    spacy dataframe result and shows the 
    average words and the average number 
    of words and the unique words for each year.
    '''
    summary_dataframe = pd.DataFrame()
    years = df['Year'].unique().tolist()
    for i in range(0, len(years)):
        row = {
            "Year": years[i],
            "Average Words": df['Word Counts'][df['Year'] == years[i]].mean(),
            "Unique Words": df['Unique Word Counts'][df['Year'] == years[i]].mean()
        }
        summary_dataframe = summary_dataframe.append(row, ignore_index=True)
    summary_dataframe["Year"] = summary_dataframe['Year'].astype(int)
    return summary_dataframe

def wordcloud(df):
    '''
    Function that creates a wordcloud plot 
    from the given DataFrame.
    '''
    comment_words = ''
    stopwords = set(STOPWORDS)

    for val in df:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
         
        comment_words += " ".join(tokens)+" "
     
    wordcloud = WordCloud(width = 800, height = 600,
                    max_words=250,
                    background_color ='white',
                    colormap='viridis',
                    collocations=False,
                    stopwords = STOPWORDS,
                    ).generate(comment_words)
     
    # plot the Word Cloud image                      
    plt.figure(figsize = (16,9), facecolor = None)
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    return plt.show()

def word_frequencies(df):
    '''
    Function that counts the frequencies
    of a word in a column.
    '''
    word_freq = pd.DataFrame()
    years = df['Year'].unique().tolist()
    for i in range (0, len(years)):
        year_corpus = str(df['Corpus'][df['Year'] == years[i]].tolist())
        tokens = year_corpus.split(" ")
        counts = Counter(tokens)
        word_freq = word_freq.append({
            "Year": years[i],
            "Most Common Words": counts.most_common(n=100)
    }, ignore_index=True)
    word_freq['Year'] = word_freq['Year'].astype(int)
    return word_freq

def map_popular_terms(df, column, year_column):
    '''
    Function that counts popular nouns, adverbs,
    verbs and words in a column.
    '''
    frequencies = pd.DataFrame()
    years = df[year_column].unique().tolist()
    for i in range (0, len(years)):
        year_corpus = str(df[column][df[year_column] == years[i]].tolist())
        tokens = year_corpus.split(" ")
        counts = Counter(tokens)
        frequencies = frequencies.append({
            "Year": years[i],
            "Most Common Terms": counts.most_common(n=100)
        }, ignore_index=True)
    frequencies['Year'] = frequencies['Year'].astype(int)
    return frequencies

# funciones para la positividad de las canciones
# volverla a probar...
def positive_words(df):
    '''
    Function that compares the words of each lyrics
    to a positive word list dictionary.
    '''
    positive_words = open("positive_words.txt","r")
    positive_words_data = positive_words.read()
    positive_words_list = positive_words_data.replace('\n', ' ').split(".")
    
    positive_words_per_song = []
    for i in range (0, len(df)):
        positive_words_found = []
        for line in positive_words_list:
            for positive_word in line.split(" "):
                if positive_word in df.iloc[i]['Corpus']:
                    positive_words_found.append(positive_word)
    positive_words_per_song.append(positive_words_found)
    return positive_words_per_song





