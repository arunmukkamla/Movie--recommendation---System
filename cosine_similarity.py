from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London","Paris Paris London"]
#find the count of each item in the above text

cv=CountVectorizer() #initialization of variable to this class


count_matrix=cv.fit_transform(text)


print(count_matrix.toarray())# prints [2,1] [1,2]
# next time is to find the cosine similarity..

similarity_scores=cosine_similarity(count_matrix)

print(similarity_scores)#[1 , 0.8]==>first is similar to first by 100% where as to second it is 50%


