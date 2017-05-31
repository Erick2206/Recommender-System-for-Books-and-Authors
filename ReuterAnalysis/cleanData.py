import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

fileName="data.csv"
cachedStopWords=stopwords.words("english")
ps=PorterStemmer()

data=pd.read_csv(fileName,delimiter="    ",engine="python")
for index,row in data.iterrows():
    word_dataset=[]
    sentences=sent_tokenize(row['Article'])
    ##words=[word for word in words if word not in cachedStopWords]
    for words in sentences:
        word=word_tokenize(words)
        for allwords in word:
            allwords=ps.stem(allwords)
            word_dataset.append(allwords)
    row['Article']=' '.join(word_dataset)

data.index.name="Index"
data.to_csv(fileName,sep='\t')
