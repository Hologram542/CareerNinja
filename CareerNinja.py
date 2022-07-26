import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def similarity(text1, text2):
    data = [text1, text2]
    vectorizer = TfidfVectorizer()
    vector_matrix = vectorizer.fit_transform(data)
    tokens = vectorizer.get_feature_names()
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    return cosine_similarity_matrix[0][1]


similarity_score = similarity("Sun is red in color", "Sky is blue in color")
print(similarity_score)
