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
        """
        with open("BagOfWords.txt", "w") as file:
            for i in BOW:
                file.write(str(i)) 
        """
        return BOW

def cosineDistance(bagOfWords):
    cosineSimilarities = linear_kernel(bagOfWords, bagOfWords)
    cosineSimilarities = cosineSimilarities.flatten()
    cosineSimilarities = cosineSimilarities[cosineSimilarities != 1]
    """
    with open("CosineDistances.txt", "w") as file:
            for i in cosineSimilarities:
                file.write(str(i))
    """
    plt.hist(cosineSimilarities, bins = 500)
    plt.show()
    
def euclideanDistance(bagOfWords):
    euclideanSimiliarties = euclidean_distances(bagOfWords)
    euclideanSimiliarties = euclideanSimiliarties.flatten()
    euclideanSimiliarties = euclideanSimiliarties[euclideanSimiliarties != 0]
    """
    with open("EuclideanDistances.txt", "w") as file:
        for i in euclideanSimiliarties:
            file.write(str(i))    
    """
    plt.hist(euclideanSimiliarties, bins = 500)
    plt.show()
    
readAndProcess("cnnhealth.txt")
bagOfWords = bagOfWordsCreator(textArray)
cosineDistance(bagOfWords)
euclideanDistance(bagOfWords)
