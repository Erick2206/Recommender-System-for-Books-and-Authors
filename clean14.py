import pandas as pd

##Outfile name for cleaned data
outfile="papers14edit.csv"
data=pd.read_csv('papers14.csv')

##Iterating contents of csv and removing unnecessary spaces
for index,row in data.iterrows():
    row["keywords"]=row["keywords"].split('\n')
    if not pd.isnull(row["topics"]):
        row["topics"]=row["topics"].split("\n")
    row["abstract"]=row["abstract"].replace('\n',' ')
    if not pd.isnull(row["groups"]):
        row['groups']=row['groups'].split("\n")

##Giving new dataframe index a name
data.index.name="index"
data.to_csv(outfile, sep=',', encoding='utf-8')
