import os

target="data.csv"
writeMode="w"
rootDir="/home/erick/Repo/InternWork/ReuterAnalysis/Sample"

with open(target,writeMode) as out:
    for dirName, subDirList, fileList in os.walk(rootDir):
        print dirName
        for fName in fileList:
            print fName
        print len(fileList)
