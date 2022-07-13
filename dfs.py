import graph_tool
import pandas as pn
import graph_tool.util
import graph_tool.topology
import graph_tool.centrality
import graph_tool.stats
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from xml.etree import cElementTree as elementTree
import re
import graph_tool
import graph_tool.util
import graph_tool.stats
from numpy.random import *  # for random sampling
import pickle

############################################
# Adapted from https://github.com/TechnionTDK/repurposing/blob/master/createSemRepGraph.py
# Describe here: @inproceedings{Nordon2019SeparatingWF,
#  title={Separating Wheat from Chaff: Joining Biomedical Knowledge and Patient Data for Repurposing Medications},
#  author={Galia Nordon and Gideon Koren and Varda Shalev and Eric Horvitz and Kira Radinsky},
#  booktitle={AAAI},
#  year={2019}
#  }
# URL: https://github.com/TechnionTDK/repurposing/blob/master/createSemRepGraph.py
############################################

############################################
# small utility functions
############################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def getPathMaxPageRank(path):
    pr = []
    for p in path:
        pr.append(pagerank[p])
    return max(pr)

def getPathNames(path):
    pathStrArry = []
    for p in path:
        txt = IdToStrDict[p]
        pathStrArry.append(txt)
    return pathStrArry

def printPageRankHist(semRepGraph):
    prarry = []
    for n in semRepGraph.nodes():
        prarry.append(pagerank[n])

semRepGraph = graph_tool.Graph()
text_property = semRepGraph.new_vertex_property("string")
semRepGraph.vertex_properties['text'] = text_property

semType_property = semRepGraph.new_vertex_property("string")
semRepGraph.vertex_properties['semType'] = semType_property

cui_property = semRepGraph.new_vertex_property("string")
semRepGraph.vertex_properties['cui'] = cui_property

predicates = pn.read_csv('~/Projects/lbd_tutorial/sempredCorpus.tsv', sep='\t')  # test2.txt is from SemRep
predicates.columns = ['PMID', 'PREDICATE', 'SUBJECT_CUI', 'SUBJECT_NAME',
                      'SUBJECT_SEMTYPE','OBJECT_CUI', 'OBJECT_NAME', 'OBJECT_SEMTYPE', 'READER']

'''
Filter data by Predicate type.
In this example, we leave only 'CAUSES'

Other predicate types include: 
 PROCESS_OF, Activation, Inhibition, LOCATION_OF, ISA, Influence, TREATS, AFFECTS, ASSOCIATED_WITH,
 COEXISTS_WITH, PREDISPOSES, PART_OF, Association, CAUSES, IncreaseAmount, USES, DecreaseAmount, INTERACTS_WITH, 
 PREVENTS, Phosphorylation, ADMINISTERED_TO, DIAGNOSES, NEG_PROCESS_OF, INHIBITS, STIMULATES, AUGMENTS, 
 DISRUPTS, MEASURES, compared_with, MANIFESTATION_OF, PRODUCES, OCCURS_IN, Dephosphorylation, PRECEDES, 
 NEG_ASSOCIATED_WITH, NEG_AFFECTS, METHOD_OF, NEG_PREDISPOSES, NEG_TREATS, NEG_COEXISTS_WITH, NEG_LOCATION_OF,
 Methylation, NEG_ISA, higher_than, NEG_ADMINISTERED_TO, NEG_PART_OF, CONVERTS_TO, NEG_INTERACTS_WITH, 
 Acetylation, Glycosylation, NEG_CAUSES, Demethylation, COMPLICATES, NEG_PREVENTS, NEG_DISRUPTS, NEG_AUGMENTS, 
 NEG_DIAGNOSES, Hydroxylation, NEG_INHIBITS, NEG_MEASURES, Deacetylation, NEG_USES, NEG_STIMULATES, 
 NEG_higher_than, NEG_OCCURS_IN, MEASUREMENT_OF, NEG_MANIFESTATION_OF, NEG_METHOD_OF, NEG_PRECEDES, 
 NEG_PRODUCES, NEG_same_as, lower_than, same_as
 
 Activation, Inhibition, Influence, TREATS, AFFECTS, PREDISPOSES, CAUSES, IncreaseAmount, DecreaseAmount, 
 INTERACTS_WITH, PREVENTS, Phosphorylation, INHIBITS, STIMULATES, AUGMENTS, DISRUPTS, Dephosphorylation, 
 Methylation, CONVERTS_TO, Acetylation, Glycosylation, Demethylation, Hydroxylation, Deacetylation 
'''


# predTypes = ['CAUSES', 'PREDISPOSES', 'Influences', 'STIMULATES', 'Activation', 'Inhibition'] #, 'Activation', 'Inhibition', 'Influence', 'AFFECTS', 'PREDISPOSES', 'IncreaseAmount', 'DecreaseAmount', 'Phosphorylation', 'INHIBITS', 'STIMULATES', 'AUGMENTS', 'DISRUPTS']

predTypes = ['TREATS', 'PREVENTS', 'Activation', 'Inhibition', 'Influence', 'CAUSES', 'IncreaseAmount', 'DecreaseAmount', 'INTERACTS_WITH', 'Phosphorylation', 'INHIBITS', 'STIMULATES', 'DISRUPTS', 'Dephosphorylation', 'Methylation',     'CONVERTS_TO', 'Acetylation', 'Glycosylation', 'Demethylation', 'Hydroxylation', 'Deacetylation', 'AUGMENTS', 'PREDISPOSES']

# predTypes = ['Activation', 'Inhibition', 'Influence', 'IncreaseAmount', 'DecreaseAmount', 'INTERACTS_WITH', 'Phosphorylation', 'INHIBITS', 'STIMULATES', 'AUGMENTS', 'DISRUPTS', 'Dephosphorylation', 'Methylation', 'CONVERTS_TO', 'Acetylation', 'Glycosylation', 'Demethylation', 'Hydroxylation', 'Deacetylation']

predicates = predicates[predicates['PREDICATE'].isin(predTypes)]

'''
Data can also be filtered by OBJECT_SEMTYPE and SUBJECT_SEMTYPE.
For example:
semtypes=['orgm','bpoc','diap','ortf','bsoj','inpo','tisu','topp','mamm','inpr','geoa','hlca','bdsy','blor','hcro','lbpr'
          ,'inbe','orga','menp','mnob','humn','amph','plnt','spco','anim','resa','anab','eehu','tmco','edac','ftcn','ocdi',
          'dora','qnco','orgt','npop','qlco','podg','prog','bird','mcha','rept','fish','phob','socb','idcn','popg',
          'bmod','emod','aggp','famg','rnlw','mbrt','pros','lang','ocac','gora','medd']
predicates = predicates[predicates['OBJECT_SEMTYPE'].isin(semtypes)==False]
predicates = predicates[predicates['SUBJECT_SEMTYPE'].isin(semtypes)==False]
'''

semtypes=['orgm','bpoc','diap','ortf','bsoj','inpo','tisu','topp','mamm','inpr','geoa','hlca','bdsy','blor','hcro','lbpr'
          ,'inbe','orga','menp','mnob','humn','amph','plnt','spco','anim','resa','anab','eehu','tmco','edac','ftcn','ocdi',
          'dora','qnco','orgt','npop','qlco','podg','prog','bird','mcha','rept','fish','phob','socb','idcn','popg',
          'bmod','emod','aggp','famg','rnlw','mbrt','pros','lang','ocac','gora','medd']
predicates = predicates[predicates['OBJECT_SEMTYPE'].isin(semtypes)==False]
predicates = predicates[predicates['SUBJECT_SEMTYPE'].isin(semtypes)==False]

removelist=['Single_Nucleotide_Polymorphism', 'multiple_pathologies', 'Natural_Products', 'TRANSCRIPTION_FACTOR', 'Transcription,_Genetic', 'Molecular_Target', 'Gene_Expression', 'Signal_Transduction', 'Homeostasis', 'Accumulation', 'Syndrome', 'Disease', 'Genetic_disorders', 'Polymorphism,_Genetic', 'Mental_disorders', 'Dementia', 'DNA', 'Alzheimer_disease,_familial,_type_3', 'Pathogenesis']
predicates = predicates[predicates['OBJECT_NAME'].isin(removelist)==False]
predicates = predicates[predicates['SUBJECT_NAME'].isin(removelist)==False]

cuiremovelist=['C0001687','C0002526','C0003043','C0003062','C0005515','C0009566','C0012634','C0013227','C0021521','C0021948','C0027361','C0027362','C0027363','C0028622','C0029224','C0029235','C0030705','C0039082','C0039796','C0087111','C0159028','C0178310','C0178341','C0178353','C0178355','C0178359','C0243192','C0422820','C0424450','C0436606','C0442826','C0476466','C0478681','C0478682','C0480773','C0481349','C0481370','C0557587','C0565657','C0580210','C0580211','C0589603','C0596048','C0596090','C0597010','C0597237','C0597240','C0677042','C0687732','C1257890','C1258127','C1318101','C1457887','C1609432','C0007634','C0020114','C0237401','C0011900','C1273869','C0449851','C0277785','C0184661','C1273870','C0185125','C0879626','C0004927','C0936012','C0311392','C0597198','C0018684','C0042567','C0029921','C0683971','C0016163','C0024660','C0687133','C0037080','C0680022','C1185740','C0871261','C0544461','C1260954','C0877248','C0242485','C0205147','C0486805','C0005839','C0021562','C0205148','C0031843','C0040223','C0205145','C0205400','C0086388','C0014406','C0520510','C0035168','C0029237','C0277784','C0001779','C0542559','C0035647','C0025664','C0700287','C0678587','C0205099','C0205146','C0237753','C0441800','C0449719','C0348026','C0008902','C0586173','C0332479','C0807955','C0559546','C0031845','C0678594','C0439792','C0557854','C1522240','C1527144','C0449234','C0542341','C0079809','C0205094','C0037455','C0025118','C0441471','C0441987','C0439534','C0392360','C0456603','C0699733','C0036397','C0725066','C0496675','C0282354','C0015127','C1273937','C1368999','C0442804','C0449286','C0205082','C0814472','C1551338','C0599883','C0450429','C1299582','C0336791','C0443177','C0025080','C1372798','C0028811','C0205246','C0449445','C0332185','C0332307','C0443228','C1516635','C0376636','C0221423','C0037778','C0199168','C0008949','C0014442','C0456387','C1265611','C0243113','C0549177','C0229962','C0600686','C1254351','C0243095','C1444647','C0033684','C0338067','C0441712','C0679607','C0808233','C1373236','C0243082','C1306673','C1524062','C0002085','C0243071','C0238767','C0005508','C0392747','C0008633','C0205195','C0205198','C0456205','C0521116','C0011155','C1527240','C1527148','C0743223','C0178602','C1446466','C0013879','C0015295','C1521761','C1522492','C0017337','C0017428','C0017431','C0079411','C0018591','C0019932','C0021149','C0233077','C0021920','C0022173','C1517945','C0680220','C0870883','C0567416','C0596988','C0243132','C0029016','C1550456','C0243123','C0030956','C0851347','C0031328','C0031327','C00314','C1514468','C0033268','C0449258','C0871161','C1521828','C0443286','C1547039','C1514873','C0035668','C0439793','C0205171','C0449438','C1547045','C0449913','C0042153','C0205419','C1441526','C1140999','C0679670','C0431085','C1185625','C1552130','C1553702','C1547020','C0242114','C0439165','C0679646','C0599755','C0681850','6275','6285']
predicates = predicates[predicates['OBJECT_CUI'].isin(cuiremovelist)==False]
predicates = predicates[predicates['SUBJECT_CUI'].isin(cuiremovelist)==False]

nodeDict = {}

subjects = predicates['SUBJECT_NAME'].unique()
objects = predicates['OBJECT_NAME'].unique()
nodes = list(set(subjects).union(set(objects)))
for n in nodes:
    newNode = semRepGraph.add_vertex()
    text_property[newNode] = n
    nodeDict[n] = int(newNode)

'''
subjectcuidf = predicates[['SUBJECT_NAME', 'SUBJECT_CUI']].copy()
subjectcuidf = subjectcuidf.drop_duplicates()
subjectcuidf = subjectcuidf.rename(columns={'SUBJECT_NAME': 'NAME', 'SUBJECT_CUI': 'CUI'})
objectcuidf = predicates[['OBJECT_NAME', 'OBJECT_CUI']].copy()
objectcuidf = objectcuidf.drop_duplicates()
objectcuidf = objectcuidf.rename(columns={'OBJECT_NAME': 'NAME', 'OBJECT_CUI': 'CUI'})
cuidf = subjectcuidf.append(objectcuidf)
'''

index = 0
edges = []
while index < len(predicates):
    #print(index)
    object = nodeDict[predicates.iloc[index]['SUBJECT_NAME']]
    subject = nodeDict[predicates.iloc[index]['OBJECT_NAME']]
    edges.append((object, subject))
    index += 1
    #print(subject)

semRepGraph.add_edge_list(edges)

graph_tool.stats.remove_self_loops(semRepGraph)

semRepGraph.save("/Users/scottalexandermalec/Projects/lbd_tutorial/semRepGraph.xml")  # save garph
save_obj(nodeDict, "/Users/scottalexandermalec/Projects/lbd_tutorial/nodeDict")   # save dictionary

# load graph from memory
semRepGraph = graph_tool.load_graph("/Users/scottalexandermalec/Projects/lbd_tutorial/semRepGraph.xml")
nodeDict = load_obj("/Users/scottalexandermalec/Projects/lbd_tutorial/nodeDict")

# init IdTpStrDict
IdToStrDict = {}
for k in nodeDict.keys():
    IdToStrDict[nodeDict[k]] = k

# calculate page rank for path filtering. Example of filtering by max page rank shown below
pagerank = graph_tool.centrality.pagerank(semRepGraph)

# get source and target nodes according to the object of subject name in UMLS

s = nodeDict['Sleep_Apnea,_Obstructive'] # Quercetin, Curcumin, Ischemic_stroke, Vitamin_D, Stroke,_Acute, Chronic_infectious_disease
t = nodeDict['Alzheimers_Disease']

# find paths from source to target
paths = graph_tool.topology.all_paths(semRepGraph, s, t, cutoff=5)

print("Searching knowledge extraction graph")

# convert paths to text paths (paths where nodes have UMLS names)
count = 0
textpaths = pn.DataFrame()
prarray = []  # will hold max page rank for each path
for p in paths:
    prarray.append(getPathMaxPageRank(p))
    textpaths = textpaths.append(pn.DataFrame([getPathNames(p)]))
    count += 1
    print(count)
    print(pn.DataFrame([getPathNames(p)]).drop_duplicates())

textpaths = textpaths.drop_duplicates()

print("####################")
print(len(textpaths))
print(textpaths.to_string())


