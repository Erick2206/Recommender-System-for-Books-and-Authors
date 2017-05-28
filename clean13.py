import pandas as pd

##Name of the outfile
outfile="papers13edit.csv"

##Read the csv file
data=pd.read_csv('paper13.csv')

##Iterate over the rows of the dataset and remove unnecessary '\n'
for index,row in data.iterrows():
    row["Keywords"]=row["Keywords"].split('\n')
    if not pd.isnull(row["Topics"]):
        row["Topics"]=row["Topics"].split("\n")
    row["Abstract"]=row["Abstract"].replace('\n',' ')
    row['High-Level Keyword(s)']=row['High-Level Keyword(s)'].split("\n")

##Set the index value for the cleaned dataset
data.index.name="index"

##Save the dataset to a new csv file
data.to_csv(outfile, sep=',', encoding='utf-8')
