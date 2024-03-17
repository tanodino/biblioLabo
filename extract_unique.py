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

def replaceSpecialCases(authors_list, hash_special_cases):
    new_list = []
    for el in authors_list:
        if el in hash_special_cases:
            new_list.append(hash_special_cases[el])
        else:
            new_list.append(el)
    return new_list


def extractAllPairs(authors, ref_names, hash_special_cases):
    authors_list = authors.split(" and ")
    authors_list = [unidecode(el).lower().strip() for el in authors_list]    
    authors_list = ["%s %s"%(el.split(" ")[0], el.split(" ")[-1]) for el in authors_list]
    authors_list = replaceSpecialCases(authors_list, hash_special_cases)
    results = np.intersect1d(authors_list, ref_names)
    pairs = list(itertools.combinations(results, 2))
    return pairs

def extractConnections(folderName, filePermanents):
    hash_name2insti = {}
    hash_name2team = {}
    df = pd.read_csv(filePermanents)
    prenom = df.iloc[:, 2].to_numpy()
    nom = df.iloc[:, 1].to_numpy()
    prenom = [ unidecode(el).lower().strip() for el in prenom]
    nom = [ unidecode(el).lower().strip() for el in nom]
    nom = [el.split(" ")[-1] for el in nom]
    institutes = df.iloc[:,-1].to_numpy()
    institutes = [el.strip() for el in institutes]

    teams = df.iloc[:,-2].to_numpy()
    teams = [el.strip() for el in teams]

    for i in range(len(nom)):
        hash_name2insti[nom[i]] = institutes[i]
    
    for i in range(len(nom)):
        hash_name2team[nom[i]] = teams[i]

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


    fNames = glob.glob(folderName+"/*.json")
    hash_pairs = {}
    for j_fileName in fNames:
        f = open(j_fileName)
        data = json.load(f)
        authors = data['bib']['author']
        pairs = extractAllPairs(authors, ref_names, hash_special_cases)
        for p in pairs:
            new_p = list(p)
            new_p.sort()
            key_pair = '%s_%s'%(new_p[0], new_p[1])
            if key_pair not in hash_pairs:
                hash_pairs[key_pair] = 0
            hash_pairs[key_pair] = hash_pairs[key_pair] + 1
        f.close()
    return hash_pairs, hash_name2insti, hash_name2team

start_date = int(sys.argv[1])
end_date = int(sys.argv[2])
edge_th = int(sys.argv[3])

hash_color = {0:'#FF0000', 1: '#00FF00', 2: '#0000FF', 3: '#FFFF00', 4: '#FF00FF', 5: '#00FFFF', 6 : '#800080', 7: '#008000'}
hash_shape = {0:'start', 1:'square', 2:'dot', 3:'triangle', 4:'diamond',5:'triangleDown',6:'icon',7:'image'}


#year_th = 2016
folderName = "full_list"
if not os.path.exists(folderName):
    os.makedirs(folderName)

hashPub = {} 
prefix_folder = "filtered"
listNames = glob.glob("*")
dir_names = [(el) for el in listNames if os.path.isdir(el) ]

for d_name in dir_names:
    path = d_name+"/"+prefix_folder
    if os.path.exists(path):
        json_files = glob.glob(path+"/*.json")
        for j_fileName in json_files:
            f = open(j_fileName)
            data = json.load(f)
            if 'pub_url' not in data.keys():
                continue
            id_pub = data['pub_url']
            if 'pub_year' not in data['bib']:
                continue
            year = int(data['bib']['pub_year'])
            if (id_pub not in hashPub) and (year >= start_date) and (year <= end_date):
                hashPub[id_pub] = data
            f.close()

count = 0
for k in hashPub.keys():
    save_file = open("%s/pub_%d.json"%(folderName,count), "w")  
    json.dump(hashPub[k], save_file, indent = 4)  
    save_file.close()  
    count+=1


G = nx.Graph()
filePermanents = 'permanents_TETIS.csv'
hash_pairs, hash_name2insti, hash_name2team = extractConnections(folderName, filePermanents)
hash_pairs = {k: v for k, v in sorted(hash_pairs.items(), key=lambda item: -item[1])}
edges = []
for k in hash_pairs:
    print("%s %d"%(k,hash_pairs[k]))
    el1, el2 = k.split("_")
    el1 = el1.split(" ")[-1]
    el2 = el2.split(" ")[-1]
    #if hash_pairs[k] > 1:
    if hash_pairs[k] >= edge_th:
        edges.append([el1,el2,hash_pairs[k]/5])



hash_insti2colors = {}
for v in hash_name2insti.values():
    if v not in hash_insti2colors:
        hash_insti2colors[v] = hash_color[len(hash_insti2colors)]

hash_team2colors = {}
for v in hash_name2team.values():
    if v not in hash_team2colors:
        hash_team2colors[v] = hash_color[len(hash_team2colors)]

hash_team2shape = {}
for v in hash_name2team.values():
    if v not in hash_team2shape:
        hash_team2shape[v] = hash_shape[len(hash_team2shape)]

hash_insti2shape = {}
for v in hash_name2insti.values():
    if v not in hash_insti2shape:
        hash_insti2shape[v] = hash_shape[len(hash_insti2shape)]

G.add_weighted_edges_from(edges)
net = Network(height='1000px')
net.from_nx(G)
for node in net.nodes:
    print(node)
    print(hash_name2insti[node['id']])
    print( hash_insti2colors[ hash_name2insti[node['id']] ] )
    #node['color'] = hash_team2colors[ hash_name2team[node['id']] ]
    #node['shape'] = hash_insti2shape[ hash_name2insti[node['id']] ]

    node['color'] = hash_insti2colors[ hash_name2insti[node['id']] ]
    node['shape'] = hash_team2shape[ hash_name2team[node['id']] ]
    
    print(node)
    print("======")


net.save_graph("network_%d_%d_%d.html"%(start_date,end_date,edge_th) )