import os

target="data.csv"
writeMode="wb"
rootDir="/home/erick/Repo/InternWork/ReuterAnalysis/Train"

with open(target,writeMode) as out:
    out.write("Author    Article\n")
    for dirName, subDirList, fileList in os.walk(rootDir):
        print dirName
        for fName in fileList:
            filePath=dirName+'/'+fName
            with open(filePath,'rb') as inFile:
                data=inFile.read().replace("\r\n","")
                data=data.replace("'","")
                data=data.replace("    "," ")
                out.write(dirName.split('/')[-1]+'    ')
                out.write(data)
                out.write('\n')
