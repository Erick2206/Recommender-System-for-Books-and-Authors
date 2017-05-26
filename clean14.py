import pandas as pd

outfile="papers14edit.csv"
data=pd.read_csv('papers14.csv')

##print data.head()
for index,row in data.iterrows():
    row["keywords"]=row["keywords"].split('\n')
    if not pd.isnull(row["topics"]):
        row["topics"]=row["topics"].split("\n")
#    print row["topics"]
    row["abstract"]=row["abstract"].replace('\n',' ')
    if not pd.isnull(row["groups"]):
        row['groups']=row['groups'].split("\n")

data.index.name="index"
data.to_csv(outfile, sep=',', encoding='utf-8')
