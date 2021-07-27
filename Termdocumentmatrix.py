# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:09:05 2020

@author: James Nelan
"""


# The below code will import the packages needed for this problem
import nltk
import re
import string
import pandas
import sklearn
import os

# This will set up the working directory
os.chdir("\\\\Desktop")

# This will import the dataset into a variable
with open("passage.txt","r") as fileWithData:
    listOfParagraphs = fileWithData.readlines()
    
# this will create a list variable to keep track of post-processed data
listOfPostProcessedParagraphs = []

# this for loop will clean the data in listOfParagraphs (i.e remove punctuation,HTML, whitespace)
for paragraph in listOfParagraphs:
    modifiedParagraph = re.sub("<.*?>", "", paragraph)
    modifiedParagraph = modifiedParagraph.translate(str.maketrans("","", string.punctuation))
    modifiedParagraph = " ".join(modifiedParagraph.split())
    
# This will create a variable to hold list of stemmed words with stop words removed
    listOfStemmedWordsWithStopWordsRemoved = []

# This for loop with contain the stemmed words as well as remove stopwords
    for word in modifiedParagraph.split(" "):
        if word not in nltk.corpus.stopwords.words("english"):
            listOfStemmedWordsWithStopWordsRemoved.append(nltk.stem.snowball.SnowballStemmer("english").stem(word))

# We will now take the listOfStemmedWordsWithStopWordsRemoved and turn it back to a paragraph
    modifiedParagraph = " ".join(listOfStemmedWordsWithStopWordsRemoved)

# this will removed grouped spaces
    modifiedParagraph = re.sub("\s\s+", " ", modifiedParagraph)

# this will add pre-processed data and move it to listOfPostProcessedParagraphs
    listOfPostProcessedParagraphs.append(modifiedParagraph)

# This will create a Term Document Matrix form listOfPostProcessedParagraphs
cv = sklearn.feature_extraction.text.CountVectorizer()
data = cv.fit_transform(listOfPostProcessedParagraphs)                  
dfTDM = pandas.DataFrame(data.toarray(), columns=cv.get_feature_names())

# This will export the term document matrix into an excel file
dfTDM.to_excel("termDocumentMatrix.xlsx", index = False)


# This will create a list to hold the words of the paragraphs
tokenizedText = []
 
# This for loop will split the sentence apart from the words
for word in " ".join(listOfPostProcessedParagraphs).split(" "):
    tokenizedText.append(word)
    
# this will generate the frequency distribution
nltk.probability.FreqDist(tokenizedText)


# This will generate the 5,10,15 most frequently used terms
nltk.probability.FreqDist(tokenizedText).most_common(5)
nltk.probability.FreqDist(tokenizedText).most_common(10)
nltk.probability.FreqDist(tokenizedText).most_common(15)


# This will generate a frequency distribution plot of 15 most common terms
nltk.probability.FreqDist(tokenizedText).plot(15)


# this will implement the lemmatization function for this dataset
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:                        
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


# this will store a new variable for post processed lemmatized data
listOfPostProcessedParagraphsTwo = []

# this for loop will clean the data in listOfParagraphs (i.e remove punctuation,HTML, whitespace)
for paragraph in listOfParagraphs:
    modifiedParagraph = re.sub("<.*?>", "", paragraph)
    modifiedParagraph = modifiedParagraph.translate(str.maketrans("","", string.punctuation))
    modifiedParagraph = " ".join(modifiedParagraph.split())
# this will lemmatize the words within the modifiedParagraph
    modifiedParagraph = lemmatize_sentence(modifiedParagraph)
    
# This will create a variable to hold list of lemmatized words with stop words removed
    listOfLemmatizedWordsWithStopWordsRemoved = []

# This for loop with contain the stemmed words as well as remove stopwords
    for word in modifiedParagraph.split(" "):
        if word.lower() not in nltk.corpus.stopwords.words("english"):
            listOfLemmatizedWordsWithStopWordsRemoved.append(word)

# We will now take the listOfStemmedWordsWithStopWordsRemoved and turn it back to a paragraph
    modifiedParagraph = " ".join(listOfLemmatizedWordsWithStopWordsRemoved)

# this will removed grouped spaces
    modifiedParagraph = re.sub("\s\s+", " ", modifiedParagraph)

# this will add pre-processed data and move it to listOfPostProcessedParagraphs
    listOfPostProcessedParagraphsTwo.append(modifiedParagraph)
    
# This will create a Term Document Matrix form listOfPostProcessedParagraphsTwo
cv = sklearn.feature_extraction.text.CountVectorizer()
dataTwo = cv.fit_transform(listOfPostProcessedParagraphsTwo)                  
dfTDMTwo = pandas.DataFrame(dataTwo.toarray(), columns=cv.get_feature_names())

# This will export the term document matrix into an excel file
dfTDMTwo.to_excel("termDocumentMatrixTwo.xlsx", index = False)


# This will create a list to hold the words of the paragraphs
tokenizedTextTwo = []
 
# This for loop will split the sentence apart from the words
for word in " ".join(listOfPostProcessedParagraphsTwo).split(" "):
    tokenizedTextTwo.append(word)
    
# this will generate the frequency distribution
nltk.probability.FreqDist(tokenizedTextTwo)


# This will generate the 5,10,15 most frequently used terms
nltk.probability.FreqDist(tokenizedTextTwo).most_common(5)
nltk.probability.FreqDist(tokenizedTextTwo).most_common(10)
nltk.probability.FreqDist(tokenizedTextTwo).most_common(15)


# This will generate a frequency distribution plot of 15 most common terms
nltk.probability.FreqDist(tokenizedTextTwo).plot(15)
