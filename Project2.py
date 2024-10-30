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

def BagOfWordsCreator(textArray):
        vectorizer = TfidfVectorizer(max_features=5000)
        BOW = vectorizer.fit_transform(textArray)     
        return BOW

def CosineDistance(BagOfWords):
    cosineSimilarities = linear_kernel(BagOfWords, BagOfWords)
    cosineSimilarities = cosineSimilarities.flatten()
    cosineSimilarities = cosineSimilarities[cosineSimilarities != 1]
    plt.hist(cosineSimilarities, bins = 500)
    plt.show()
    
readAndProcess("cnnhealth.txt")
BagOfWords = BagOfWordsCreator(textArray)
CosineDistance(BagOfWords)