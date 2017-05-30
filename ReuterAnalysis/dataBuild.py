import os

target="data.csv"
writeMode="wb"
rootDir="/home/erick/Repo/InternWork/ReuterAnalysis/Sample"

with open(target,writeMode) as out:
    for dirName, subDirList, fileList in os.walk(rootDir):
        print dirName
        for fName in fileList:
            filePath=dirName+'/'+fName
            with open(filePath,'rb') as inFile:
                data=inFile.read().replace("\r\n","")
                out.write(dirName.split('/')[-1]+',')
                out.write(data)
                out.write('\n')
