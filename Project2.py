import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel

textArray = []

def readAndProcess(fileName):
    with open(fileName, 'r', encoding="utf8") as file:
        for line in file:
            textArray.append(line.strip().split('|')[2])

def bagOfWordsCreator(textArray):
        vectorizer = CountVectorizer()
        vectorizer.fit(textArray)
        BOW = vectorizer.transform(textArray)     
        print(BOW[0])
        return BOW

def cosineDistance(bagOfWords):
    cosineSimilarities = linear_kernel(bagOfWords, bagOfWords)
    cosineSimilarities = cosineSimilarities.flatten()
    cosineSimilarities = cosineSimilarities[cosineSimilarities != 1]
    plt.hist(cosineSimilarities, bins = 500)
    plt.show()
    
def euclideanDistance(bagOfWords):
    euclideanSimiliarties = euclidean_distances(bagOfWords)
    euclideanSimiliarties = euclideanSimiliarties.flatten()
    euclideanSimiliarties = euclideanSimiliarties[euclideanSimiliarties != 0]
    print(euclideanSimiliarties)
    plt.hist(euclideanSimiliarties, bins = 500)
    plt.show()
    
readAndProcess("cnnhealth.txt")
bagOfWords = bagOfWordsCreator(textArray)
