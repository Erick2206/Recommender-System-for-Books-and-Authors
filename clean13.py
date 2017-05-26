import pandas as pd

outfile="papers13edit.csv"
data=pd.read_csv('paper13.csv')

##print data.head()
for index,row in data.iterrows():
    row["Keywords"]=row["Keywords"].split('\n')
    if not pd.isnull(row["Topics"]):
        row["Topics"]=row["Topics"].split("\n")
#    print row["topics"]
    row["Abstract"]=row["Abstract"].replace('\n',' ')
    row['High-Level Keyword(s)']=row['High-Level Keyword(s)'].split("\n")

data.index.name="index"
data.to_csv(outfile, sep=',', encoding='utf-8')
