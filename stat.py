import glob
import os
import json
from unidecode import unidecode
import pandas as pd
import numpy as np
import itertools
from pyvis.network import Network
import networkx as nx
import math
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def getTeamsFromAuthors(authors_list, hash_rn2team):
    results = []
    for el in authors_list:
        if el in hash_rn2team:
            results.append( hash_rn2team[el] )
    results = np.array(results)
    return np.unique(results)

def replaceSpecialCases(authors_list, hash_special_cases):
    new_list = []
    for el in authors_list:
        if el in hash_special_cases:
            new_list.append(hash_special_cases[el])
        else:
            new_list.append(el)
    return new_list

def extractTETISagents(authors, ref_names, hash_special_cases):
    authors_list = authors.split(" and ")
    authors_list = [unidecode(el).lower().strip() for el in authors_list]    
    authors_list = ["%s %s"%(el.split(" ")[0], el.split(" ")[-1]) for el in authors_list]
    authors_list = replaceSpecialCases(authors_list, hash_special_cases)
    results = np.intersect1d(authors_list, ref_names)
    return results

filePermanents = 'permanents_TETIS.csv'
folderName = "full_list"
start_date = int(sys.argv[1])
end_date = int(sys.argv[2])

teams_labels = ["ATTOS", "MISCA", "USIG"]
palette = ['r', 'g', 'b']

OUTSIDE = "EXTERNAL"
OTHER_TEAMS = "OTHER TEAMS"
INTERNAL = "INTERNAL"

hash_teams_interactions = {}
for v in teams_labels:
    hash_teams_interactions[v] = {}
    hash_teams_interactions[v][INTERNAL] = 0
    hash_teams_interactions[v][OUTSIDE] = 0
    hash_teams_interactions[v][OTHER_TEAMS] = 0


df = pd.read_csv(filePermanents)
prenom = df.iloc[:, 2].to_numpy()
nom = df.iloc[:, 1].to_numpy()

prenom = [ unidecode(el).lower().strip() for el in prenom]
nom = [ unidecode(el).lower().strip() for el in nom]
nom = [el.split(" ")[-1] for el in nom]
ref_names = [ "%s %s"%(unidecode(p).lower().strip(),unidecode(n).lower().strip()) for p,n in zip(prenom,nom)]
hash_special_cases = {}
for el in ref_names:
    p, n = el.split(" ")
    val = "%s %s"%(p[0], n)
    hash_special_cases[val] = el

hash_special_cases["j-b feret"] = "jean-baptiste feret"
hash_special_cases["jb feret"] = "jean-baptiste feret"
hash_special_cases["pm villar"] = "patricio villar"
hash_special_cases["begue agnes"] = "agnes begue"
hash_special_cases["gaetano raffaele"] = "raffaele gaetano"
hash_special_cases["dino lenco"] = "dino ienco"

institutes = df.iloc[:,-1].to_numpy()
institutes = [el.strip() for el in institutes]

teams = df.iloc[:,-2].to_numpy()
teams = [el.strip() for el in teams]


hash_rn2team = {}
for i in range(len(ref_names)):
    hash_rn2team[ref_names[i]] = teams[i]


hash_rnInt = {}
hash_rnExt = {}

fNames = glob.glob(folderName+"/*.json")
team_interactions = []
for j_fileName in fNames:
    f = open(j_fileName)
    data = json.load(f)
    authors_field = data['bib']['author']
    year = int(data['bib']['pub_year'])
    if year <= start_date or year >= end_date:
        continue
    #Only one member of TETIS is in the publication
    authors = extractTETISagents(authors_field, ref_names, hash_special_cases)
    #print(authors)
    if len(authors) == 1:
        member = authors[0]
        if member not in hash_rnExt:
            hash_rnExt[member] = 0
        hash_rnExt[member] = hash_rnExt[member] + 1
        temp_team = hash_rn2team[member]
        if temp_team in hash_teams_interactions:
            hash_teams_interactions[temp_team][OUTSIDE] = hash_teams_interactions[hash_rn2team[member]][OUTSIDE] + 1

    elif len(authors) > 1:
        for member in authors:
            if member not in hash_rnInt:
                hash_rnInt[member] = 0
            hash_rnInt[member]+=1
        temp_team_list = getTeamsFromAuthors(authors, hash_rn2team)
        if len(temp_team_list) == 1:
            temp_team = temp_team_list[0]
            hash_teams_interactions[temp_team][INTERNAL] = hash_teams_interactions[temp_team][INTERNAL] + 1
        else:
            for t in temp_team_list:
                hash_teams_interactions[temp_team][OTHER_TEAMS] = hash_teams_interactions[temp_team][OTHER_TEAMS] + 1


total_keys = np.union1d(np.array(list(hash_rnExt.keys())), np.array(list(hash_rnInt.keys())) )
hash_team2ratio = {}
for k in total_keys:
    count_ext = 0
    count_int = 0
    ratio = 0
    if k in hash_rnExt:
        count_ext = hash_rnExt[k]
    if k in hash_rnInt:
        count_int = hash_rnInt[k]
    ratio = float(count_int) / (count_ext + count_int)
    current_team = hash_rn2team[k]
    if current_team not in hash_team2ratio:
        hash_team2ratio[current_team] = []
    hash_team2ratio[current_team].append(ratio)
    print("%s %f"%(k, ratio))

for t in hash_team2ratio:
    print("%s %f"%(t, np.mean(hash_team2ratio[t])))



'''
##### VISUALIZATION INTRA-INTER ######
vals, names, xs = [],[],[]
for i in range(len(teams_labels)):
    vals.append(hash_team2ratio[teams_labels[i]])
    names.append(teams_labels[i])
    xs.append(np.random.normal(i + 1, 0.04, len(hash_team2ratio[teams_labels[i]]) ))

plt.boxplot(vals, labels=names)
plt.title("Intra Lab Publication Ratio %d %d"%(start_date, end_date))
plt.ylabel("INTRA RATIO")
for x, val, c in zip(xs, vals, palette):
    plt.scatter(x, val, alpha=0.4, color=c)
plt.savefig("intra_inter_%d_%d.png"%(start_date,end_date))
##### VISUALIZATION INTRA-INTER ######
'''


matrix = []
for k in teams_labels:
    temp = []
    for val in [OUTSIDE, INTERNAL, OTHER_TEAMS]:
        temp.append( hash_teams_interactions[k][val] )
    matrix.append(temp)
matrix = np.array(matrix)


#### EXTERIOR / INTERIOR TEAM PUBLICATION RATIO #########
print(matrix)
normMatrix1 = matrix / matrix.sum(axis=1,keepdims=True)
print(normMatrix1)

width = 0.2
x = np.arange(normMatrix1.shape[0])

plt.title("EXTERIOR/INTERIOR LAB Pubblication Ratio %d %d"%(start_date, end_date))
plt.bar(x-0.2, normMatrix1[:,0], width, color=palette[0]) 
plt.bar(x, normMatrix1[:,1], width, color=palette[1]) 
plt.bar(x+0.2, normMatrix1[:,2], width, color=palette[2]) 
plt.xticks(x, teams_labels) 
plt.xlabel("Teams") 
plt.ylabel("Ratio") 
plt.legend([OUTSIDE, INTERNAL, OTHER_TEAMS]) 
#plt.show() 
plt.savefig("exterior_interior_lab_pub_ratio_%d_%d.png"%(start_date,end_date))
#### PROCESSING AND ANALYZING TEAM INTERACTIONS #########

plt.clf()

#### EXTERIOR / INTERIOR TEAM PUBLICATION ABS #########
width = 0.2
x = np.arange(matrix.shape[0])

plt.title("EXTERIOR/INTERIOR LAB Pubblication ABS %d %d"%(start_date, end_date))
plt.bar(x-0.2, matrix[:,0], width, color=palette[0]) 
plt.bar(x, matrix[:,1], width, color=palette[1]) 
plt.bar(x+0.2, matrix[:,2], width, color=palette[2]) 
plt.xticks(x, teams_labels) 
plt.xlabel("Teams") 
plt.ylabel("# Pub.") 
plt.legend([OUTSIDE, INTERNAL, OTHER_TEAMS]) 
#plt.show() 
plt.savefig("exterior_interior_lab_pub_abs_%d_%d.png"%(start_date,end_date))
#### PROCESSING AND ANALYZING TEAM INTERACTIONS #########

plt.clf()

#### INTERIOR TEAM PUBLICATION RATIO #########
normMatrix2 = matrix[:,1::] / matrix[:,1::].sum(axis=1,keepdims=True)
print(normMatrix2)

width = 0.2
x = np.arange(normMatrix2.shape[0])

plt.title("INTERIOR LAB Pubblication Ratio %d %d"%(start_date, end_date))
plt.bar(x-0.2, normMatrix2[:,0], width, color=palette[1]) 
plt.bar(x, normMatrix2[:,1], width, color=palette[2]) 
plt.xticks(x, teams_labels) 
plt.xlabel("Teams") 
plt.ylabel("Ratio")
plt.legend([INTERNAL, OTHER_TEAMS]) 
#plt.show() 
plt.savefig("interior_lab_pub_ratio_%d_%d.png"%(start_date,end_date))
#### PROCESSING AND ANALYZING TEAM INTERACTIONS #########