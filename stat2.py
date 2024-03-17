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
palette = ['r', 'g', 'b','c']

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

###### # PUBBLICATION PER YEAR WHOLE LAB #####
hash_year = {}
for j_fileName in fNames:
    f = open(j_fileName)
    data = json.load(f)
    authors_field = data['bib']['author']
    year = int(data['bib']['pub_year'])
    if year >= start_date and year <= end_date:
        if year not in hash_year:
            hash_year[year] = 0
        hash_year[year] = hash_year[year] + 1

vec_year = []
labels_year = []
for i in range(start_date,end_date+1):
    print("%d %d"%(i, hash_year[i]))
    vec_year.append(hash_year[i])
    labels_year.append(i)

width = 0.2
x = np.arange(len(labels_year))

plt.title("NUM PUBBLICATION PER YEAR %s %s"%(start_date,end_date))
plt.bar(x, vec_year, width*3) 
plt.xticks(x, labels_year) 
plt.xlabel("Year") 
plt.ylabel("Num Publication") 
#plt.show() 
plt.savefig("num_publication_total_%d_%d.png"%(start_date,end_date))

##########################################
plt.clf()

###### # PUBBLICATION PER YEAR PER TEAM #####
hash_team_year = {}
for j_fileName in fNames:
    f = open(j_fileName)
    data = json.load(f)
    authors_field = data['bib']['author']
    year = int(data['bib']['pub_year'])
    if year >= start_date and year <= end_date:
        authors = extractTETISagents(authors_field, ref_names, hash_special_cases)    
        for member in authors:
            temp_team = hash_rn2team[member]
            print("|%s|"%temp_team)
            if temp_team not in teams_labels:
                continue
            if temp_team not in hash_team_year:
                hash_team_year[temp_team] = {}
            if year not in hash_team_year[temp_team]:
                hash_team_year[temp_team][year] = 0
            hash_team_year[temp_team][year] = hash_team_year[temp_team][year] + 1

width = 0.2

count_team = []
for el in teams_labels:
    temp_vec = []
    for i in range(start_date,end_date+1):
        #print("i %d"%i)
        if i in hash_team_year[el]:
            print("%d %s %d"%(i,el,hash_team_year[el][i]))
            temp_vec.append( hash_team_year[el][i])
        else:
            temp_vec.append(0)
    count_team.append(temp_vec)

count_team = np.array(count_team)
print(count_team)
print(labels_year)

plt.title("NUM PUBBLICATION PER YEAR PER TEAM %s %s"%(start_date,end_date))
plt.bar(x-0.2, count_team[0], width, color=palette[0]) 
plt.bar(x, count_team[1], width, color=palette[1]) 
plt.bar(x+0.2, count_team[2], width, color=palette[2]) 
plt.xticks(x, labels_year) 
plt.xlabel("Year") 
plt.ylabel("Num Publication") 
plt.legend(teams_labels) 
#plt.show() 
plt.savefig("num_publication_team_%d_%d.png"%(start_date,end_date))

##########################################
plt.clf()
###### # 25th, 50th, 75th and average publication count PER TEAM ####
hash_team_member_count = {}
for j_fileName in fNames:
    f = open(j_fileName)
    data = json.load(f)
    authors_field = data['bib']['author']
    year = int(data['bib']['pub_year'])
    if year >= start_date and year <= end_date:
        authors = extractTETISagents(authors_field, ref_names, hash_special_cases)
        for member in authors:
            temp_team = hash_rn2team[member]
            if temp_team in teams_labels:
                if temp_team not in hash_team_member_count:
                    hash_team_member_count[temp_team] = {}
                if member not in hash_team_member_count[temp_team]:
                    hash_team_member_count[temp_team][member] = 0
                hash_team_member_count[temp_team][member] = hash_team_member_count[temp_team][member] + 1

counts_distrib_teams = []
for v in teams_labels:
    current_counts = []
    for k in hash_team_member_count[v]:
        current_counts.append(hash_team_member_count[v][k])
    counts_distrib_teams.append([np.percentile(current_counts,25),np.percentile(current_counts,50), np.percentile(current_counts,75), np.mean(current_counts) ])

counts_distrib_teams = np.array(counts_distrib_teams)

width = 0.2
x = np.arange( len(teams_labels) )
plt.title("STAT PUB PER TEAM (ONLY ACTIVE PUBLISHER) %s %s"%(start_date,end_date))
plt.bar(x-0.4, counts_distrib_teams[:,0], width, color=palette[0]) 
plt.bar(x-0.2, counts_distrib_teams[:,1], width, color=palette[1]) 
plt.bar(x, counts_distrib_teams[:,2], width, color=palette[2]) 
plt.bar(x+0.2, counts_distrib_teams[:,3], width, color=palette[3]) 
print(teams_labels)
plt.xticks(x, teams_labels) 
#plt.xlabel("Year") 
plt.ylabel("Num Publication") 
plt.legend(["25th","50th","75th","avg"]) 
#plt.show() 
plt.savefig("publication_team_stat_%d_%d.png"%(start_date,end_date))
