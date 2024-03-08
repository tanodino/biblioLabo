import glob
import os
import shutil
import json


def checkIfJorC(file_name):
    f = open(file_name)
    data = json.load(f)
    if 'conference' in data['bib'].keys() or 'journal' in data['bib'].keys():
        return True
    else:
        return False


def filter(dir_name, prefix_folder):
    newInnerDir = dir_name+"/"+prefix_folder
    if not os.path.exists(newInnerDir):
        os.makedirs(newInnerDir)
    jsonFiles = glob.glob(dir_name+"/*.json")
    for jsonF in jsonFiles:
        if checkIfJorC(jsonF):
            shutil.copy(jsonF, newInnerDir)


prefix_folder = "filtered"
fileList = glob.glob("*")
dir_names = [(el) for el in fileList if os. path.isdir(el) ]
for dir_n in dir_names:
    filter(dir_n, prefix_folder)

for dir_n in dir_names:
    l1 = glob.glob(dir_n+"/*.json")
    if len(l1) == 0:
        continue
    l2 = glob.glob(dir_n+"/"+prefix_folder+"/*.json")
    print("%s %d %d"%(dir_n, len(l1), len(l2)))
