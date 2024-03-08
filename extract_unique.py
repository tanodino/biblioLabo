import glob
import os
import json
from unidecode import unidecode
import pandas as pd
import numpy as np
import itertools
from pyvis.network import Network
import networkx as nx

def extractAllPairs(authors, ref_names):
    authors_list = authors.split(" and ")
    authors_list = [unidecode(el).lower().strip() for el in authors_list]    
    authors_list = ["%s %s"%(el.split(" ")[0], el.split(" ")[-1]) for el in authors_list]
    results = np.intersect1d(authors_list, ref_names)
    pairs = list(itertools.combinations(results, 2))
    return pairs

def extractConnections(folderName, filePermanents):
    df = pd.read_csv(filePermanents)
    prenom = df.iloc[:, 2].to_numpy()
    nom = df.iloc[:, 1].to_numpy()
    prenom = [ unidecode(el).lower().strip() for el in prenom]
    nom = [ unidecode(el).lower().strip() for el in nom]
    nom = [el.split(" ")[-1] for el in nom]
    ref_names = [ "%s %s"%(unidecode(p).lower().strip(),unidecode(n).lower().strip()) for p,n in zip(prenom,nom)]
    fNames = glob.glob(folderName+"/*.json")
    hash_pairs = {}
    for j_fileName in fNames:
        f = open(j_fileName)
        data = json.load(f)
        authors = data['bib']['author']
        pairs = extractAllPairs(authors, ref_names)
        for p in pairs:
            new_p = list(p)
            new_p.sort()
            key_pair = '%s_%s'%(new_p[0], new_p[1])
            if key_pair not in hash_pairs:
                hash_pairs[key_pair] = 0
            hash_pairs[key_pair] = hash_pairs[key_pair] + 1
        f.close()
    return hash_pairs



year_th = 2015
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
            if (id_pub not in hashPub) and (year >= year_th):
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
hash_pairs = extractConnections(folderName, filePermanents)
hash_pairs = {k: v for k, v in sorted(hash_pairs.items(), key=lambda item: -item[1])}
edges = []
for k in hash_pairs:
    print("%s %d"%(k,hash_pairs[k]))
    el1, el2 = k.split("_")
    el1 = el1.split(" ")[-1]
    el2 = el2.split(" ")[-1]
    edges.append([el1,el2,hash_pairs[k]])

G.add_weighted_edges_from(edges)
net = Network()
net.from_nx(G)
net.save_graph("prova_%d.html"%year_th)