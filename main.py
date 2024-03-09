from scholarly import scholarly
from scholarly import ProxyGenerator
import time
import os
import sys
import glob
from unidecode import unidecode
import pandas as pd
import json

pg = ProxyGenerator()
scholarly.use_proxy(pg)

def retrieveWritePublications(prefix_path, author_name):
    search_query = scholarly.search_author(author_name)
    first_author_result = next(search_query)
    author = scholarly.fill(first_author_result )
    i=0
    for el in author['publications']:
        publication_filled = scholarly.fill(el)
        save_file = open("%s/pub_%d.json"%(prefix_path,i), "w")  
        json.dump(publication_filled, save_file, indent = 4)  
        save_file.close()  
        i+=1

def writeOnFile(full_list, fileName):
    with open(fileName, 'w') as f:
        for el in full_list:
            f.write(el+'\n')

def checkIfAuthorExists(author_name):
    search_query = scholarly.search_author(author_name)
    scholar_id = None
    author_name = ""
    try: # check if author is listed in google scholar
        author_stats = next(search_query)
        author_record = scholarly.fill(author_stats)
        scholar_id = author_record['scholar_id']
        print(scholar_id)
        author_name = author_record['name'].lower()
    except:
        scholar_id = 'None'
    return scholar_id, unidecode(author_name)



#READ DATA FROM CSV FILE
df = pd.read_csv('permanents_TETIS.csv')
prenom = df.iloc[:, 2].to_numpy()
nom = df.iloc[:, 1].to_numpy()
query1 = [ "\"%s %s\""%(unidecode(p).lower().strip(),unidecode(n).lower().strip()) for p,n in zip(prenom,nom)]
query2 = [ "\"%s %s\""%(unidecode(n).lower().strip(),unidecode(p).lower().strip()) for p,n in zip(prenom,nom)]

query1 = ['"agnes begue"']
query2 = ['"begue agnes"']#['"lebourgeois valentine"','"gaetano raffaele"']

#query = zip(['"valentine lebourgeois"'], ['"lebourgeois valentine"'])
query = zip(query1, query2)
for q1, q2 in query:
    scholar_id_1, author_name_1 = checkIfAuthorExists(q1)
    scholar_id_2, author_name_2 = checkIfAuthorExists(q2)
    q1 = q1.replace("\"","")
    q2 = q2.replace("\"","")
    same_author_1 = (q1 == author_name_1) 
    print("%s == %s  %d"%(q1, author_name_1, int(q1 == author_name_1) ))

    prefix_path = q1.replace(" ","_")
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)

    same_author_2 = None
    if not same_author_1:
        same_author_2 = (q2 == author_name_2)
        print("%s == %s  %d"%(q2, author_name_2, int(q2 == author_name_2) ))
    
    
    if same_author_1:
        print("%s ; %s"%(q1, scholar_id_1))
        retrieveWritePublications(prefix_path, "\""+q1+"\"")
    elif same_author_2:
        print("%s ; %s"%(q1, scholar_id_2))
        retrieveWritePublications(prefix_path, "\""+q2+"\"")
    else:
        print("%s ; %s"%(q1, "None"))
    time.sleep(10)
    sys.stdout.flush()

'''
##### FOR AGNES BEGUE #########
    
prefix_path = "agnes_begue"
if not os.path.exists(prefix_path):
    os.makedirs(prefix_path)
author = scholarly.search_author_id('ecx11oYAAAAJ')
author = scholarly.fill(author )
print(len(author['publications']))
i=0
for el in author['publications']:
    print("pub %d"%i)
    publication_filled = scholarly.fill(el)
    save_file = open("%s/pub_%d.json"%(prefix_path,i), "w")  
    json.dump(publication_filled, save_file, indent = 4)  
    save_file.close()  
    i+=1
'''