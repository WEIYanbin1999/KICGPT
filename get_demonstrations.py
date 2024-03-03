from collections import defaultdict
import time

import argparse
import json

id2entity_name = defaultdict(str)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
args = parser.parse_args()


with open('dataset/' + args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
    entity_lines = file.readlines()
    for line in entity_lines:
        _name, _id = line.strip().split("\t")
        id2entity_name[int(_id)] = _name

id2relation_name = defaultdict(str)

with open('dataset/' + args.dataset + '/get_neighbor/relation2id.txt', 'r') as file:
    relation_lines = file.readlines()
    for line in relation_lines:
        _name, _id = line.strip().split("\t")
        id2relation_name[int(_id)] = _name

train_triplet = []
test_triplet = []

for line in open('dataset/' + args.dataset + '/get_neighbor/train2id.txt', 'r'):
    head, relation, tail = line.strip('\n').split()
    train_triplet.append(list((int(head), int(relation), int(tail))))

for line in open('dataset/' + args.dataset + '/get_neighbor/test2id.txt', 'r'):
    head, relation, tail = line.strip('\n').split()
    # train_triplet.append(list((int(head), int(relation), int(tail))))
    test_triplet.append(list((int(head), int(relation), int(tail))))

for line in open('dataset/'+args.dataset+'/get_neighbor/valid2id.txt', 'r'):
    head, relation, tail = line.strip('\n').split()
    train_triplet.append(list((int(head), int(relation), int(tail))))


graph = {}
reverse_graph = {}

def init_graph(graph_triplet):

        for triple in graph_triplet:
            head = triple[0]
            rela = triple[1]
            tail = triple[2]

            if(head not in graph.keys()):
                graph[head] = {}
                graph[head][tail] = rela
            else:
                graph[head][tail] = rela

            if(tail not in reverse_graph.keys()):
                reverse_graph[tail] = {}
                reverse_graph[tail][head] = rela
            else:
                reverse_graph[tail][head] = rela
        
        # return graph, reverse_graph, node_indegree, node_outdegree

init_graph(train_triplet)



import random

def random_delete(triplet, reserved_num): 
    reserved = random.sample(triplet, reserved_num)
    return reserved

def get_onestep_neighbors(graph, source, num_samples, reverse):
    triplet = []
    try:
        nei = list(graph[source].keys())
        # nei = random.sample(graph[source].keys(), sample_num)
        if reverse == 1:
            triplet = [tuple((nei[i], graph[source][nei[i]], source)) for i in range(len(nei))]
        else:
            triplet = [tuple((source, graph[source][nei[i]], nei[i])) for i in range(len(nei))]
    except KeyError:
        pass
    except ValueError:
        nei = list(graph[source].keys())
        triplet = [tuple((source, graph[source][nei[i]], nei[i])) for i in range(len(nei))]
    return triplet

def get_entity_neighbors(traget_entity, max_triplet):

    as_head_neighbors = get_onestep_neighbors(graph, traget_entity, max_triplet // 2, 0)
    as_tail_neighbors = get_onestep_neighbors(reverse_graph, traget_entity, max_triplet // 2, 1)

    all_triplet = as_head_neighbors + as_tail_neighbors

    return all_triplet

def get_one_hop_triplet(triplet):
    head_entity = triplet[0]
    tail_entity = triplet[2]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))

    # based on the input length, tuning the max_triplet
    max_triplet = 1000000000
    head_triplet = get_entity_neighbors(head_entity, max_triplet)
    tail_triplet = get_entity_neighbors(tail_entity, max_triplet)

    temp_triplet = list(set(head_triplet + tail_triplet))
    temp_triplet = list(set(temp_triplet) - set([triplet]))


    return temp_triplet


def get_all_relation_triplet(triplet):
    relation = triplet
    all_relation_triplet = []
    for triple in train_triplet:
        if triple[1] == relation:
            all_relation_triplet.append(tuple((triple[0],triple[1],triple[2])))
    for triple in test_triplet:
        if triple[1] == relation:
            all_relation_triplet.append(tuple((triple[0],triple[1],triple[2])))
    return all_relation_triplet

def get_relation_triplet(triplet):
    relation = triplet[1]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))
    relation_triplet = []
    for triple in train_triplet:
        if triple[1] == relation:
            relation_triplet.append(tuple((triple[0],triple[1],triple[2])))
    relation_triplet = list(set(relation_triplet) - set([triplet]))
    return relation_triplet


def get_samelink_triplet_tail(triplet):
    head_entity = triplet[0]
    relation = triplet[1]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))
    samelink_triplet = []
    for triple in train_triplet:
        if triple[1] == relation and triple[0] == head_entity:
            samelink_triplet.append(tuple((triple[0],triple[1],triple[2])))
    samelink_triplet = list(set(samelink_triplet) - set([triplet]))
    return samelink_triplet
    
def get_samelink_triplet_head(triplet):
    tail_entity = triplet[2]
    relation = triplet[1]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))
    samelink_triplet = []
    for triple in train_triplet:
        if triple[1] == relation and triple[2] == tail_entity:
            samelink_triplet.append(tuple((triple[0],triple[1],triple[2])))
    samelink_triplet = list(set(samelink_triplet) - set([triplet]))
    return samelink_triplet
    
def get_samelink_id_tail(triplet):
    head_entity = triplet[0]
    relation = triplet[1]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))
    samelink_triplet = []
    for triple in train_triplet:
        if triple[1] == relation and triple[0] == head_entity:
            samelink_triplet.append(triple[2])
    samelink_triplet = list(set(samelink_triplet) - set([triplet[2]]))
    return samelink_triplet

def get_samelink_id_head(triplet):
    tail_entity = triplet[2]
    relation = triplet[1]
    triplet = tuple((triplet[0], triplet[1], triplet[2]))
    samelink_triplet = []
    for triple in train_triplet:
        if triple[1] == relation and triple[2] == tail_entity:
            samelink_triplet.append(triple[0])
    samelink_triplet = list(set(samelink_triplet) - set([triplet[0]]))
    return samelink_triplet
    


import copy


def change_(triplet_list):
    tri_text = []
    for item in triplet_list:
        # text = id2entity_name[item[0]] + '\t' + id2relation_name[item[1]] + '\t' + id2entity_name[item[2]]
        h = id2entity_name[item[0]]
        r = id2relation_name[item[1]]
        t = id2entity_name[item[2]]
        tri_text.append([h, r, t])
    return tri_text

mask_idx = 100000000
demonstrations_T_h = defaultdict(list)
demonstrations_T_t = defaultdict(list)
demonstrations_T_r_query_tail = defaultdict(list)
demonstrations_T_r_query_head = defaultdict(list)
demonstrations_T_link_base_tail = defaultdict(list)
demonstrations_T_link_base_head = defaultdict(list)
demonstrations_T_link_base_id_tail = defaultdict(list)
demonstrations_T_link_base_id_head = defaultdict(list)
all_r_triples = defaultdict(list)

test_questions = []
ID = 0
for triplet in test_triplet:
    tail_masked = copy.deepcopy(triplet)
    head_masked = copy.deepcopy(triplet)
    no_masked = copy.deepcopy(triplet)
    
    current_test_triple = defaultdict(str)
    current_test_triple['ID'] = ID
    ID = ID + 1
    current_test_triple['HeadEntity'] = id2entity_name[no_masked[0]]
    current_test_triple['Answer'] = id2entity_name[no_masked[2]]
    current_test_triple['Question'] = id2relation_name[no_masked[1]]
    test_questions.append(current_test_triple)
    tail_masked[2] = mask_idx
    head_masked[0] = mask_idx
    demonstrations_T_h['\t'.join([id2entity_name[triplet[0]], id2relation_name[triplet[1]]])] = change_(get_one_hop_triplet(tail_masked))
    demonstrations_T_t['\t'.join([id2entity_name[triplet[2]], id2relation_name[triplet[1]]])] = change_(get_one_hop_triplet(head_masked))
    demonstrations_T_r_query_tail['\t'.join([id2entity_name[triplet[0]], id2relation_name[triplet[1]]])] = change_(get_relation_triplet(tail_masked))
    demonstrations_T_r_query_head['\t'.join([id2entity_name[triplet[2]], id2relation_name[triplet[1]]])] = change_(get_relation_triplet(head_masked))
    
    T_link_base_tail = get_samelink_triplet_tail(tail_masked)
    demonstrations_T_link_base_tail['\t'.join([id2entity_name[triplet[0]], id2relation_name[triplet[1]]])] = change_(T_link_base_tail)
    
    T_link_base_head = get_samelink_triplet_head(head_masked)
    demonstrations_T_link_base_head['\t'.join([id2entity_name[triplet[2]], id2relation_name[triplet[1]]])] = change_(T_link_base_head)
    
    
    T_link_base_id_tail = get_samelink_id_tail(tail_masked)
    demonstrations_T_link_base_id_tail['\t'.join([str(triplet[0]), str(triplet[1])])] = T_link_base_id_tail
    
    T_link_base_id_head = get_samelink_id_head(head_masked)
    demonstrations_T_link_base_id_head['\t'.join([str(triplet[2]), str(triplet[1])])] = T_link_base_id_head

for key in id2relation_name:
    all_r_triples[id2relation_name[key]] = change_(get_all_relation_triplet(key))
   


# Demonstration Pools
with open("dataset/" + args.dataset + "/demonstration/tail_supplement.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_h, indent=1))

with open("dataset/" + args.dataset + "/demonstration/head_supplement.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_t, indent=1))

with open("dataset/" + args.dataset + "/demonstration/tail_analogy.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_r_query_tail, indent=1))

with open("dataset/" + args.dataset + "/demonstration/head_analogy.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_r_query_head, indent=1))


# Other support files
with open("dataset/" + args.dataset + "/demonstration/T_link_base_head.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_link_base_head, indent=1))

with open("dataset/" + args.dataset + "/demonstration/T_link_base_tail.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_link_base_tail, indent=1))

with open("dataset/" + args.dataset + "/demonstration/all_r_triples.txt", "w") as file:
    file.write(json.dumps(all_r_triples, indent=1))

with open("dataset/" + args.dataset + "/link_base_id_tail.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_link_base_id_tail, indent=1))

with open("dataset/" + args.dataset + "/link_base_id_head.txt", "w") as file:
    file.write(json.dumps(demonstrations_T_link_base_id_head, indent=1)) 
    
with open("dataset/" + args.dataset + "/test_answer.txt", "w") as file:
    file.write(json.dumps(test_questions, indent=1))


# Filter setting
from collections import defaultdict 
all_answer_head_raw = defaultdict(list)
all_answer_tail_raw = defaultdict(list)
all_answer_head = defaultdict(list)
all_answer_tail = defaultdict(list)
total_triplet = train_triplet + test_triplet
for triplet in total_triplet:
    head_ = id2entity_name[triplet[0]]
    tail_ = id2entity_name[triplet[2]]
    relation_ = id2relation_name[triplet[1]]
    all_answer_tail_raw['\t'.join([head_, relation_])].append(tail_)
    all_answer_head_raw['\t'.join([tail_, relation_])].append(head_)
for triplet in test_triplet:
    head_ = id2entity_name[triplet[0]]
    tail_ = id2entity_name[triplet[2]]
    relation_ = id2relation_name[triplet[1]]
    all_answer_tail['\t'.join([head_, relation_])] = all_answer_tail_raw['\t'.join([head_, relation_])]
    all_answer_head['\t'.join([tail_, relation_])] = all_answer_head_raw['\t'.join([tail_, relation_])]
with open("dataset/" + args.dataset + "/filter_head.txt",'w') as load_f:
    load_f.write(json.dumps(all_answer_head, indent=1))
with open("dataset/" + args.dataset + "/filter_tail.txt",'w') as load_f:
    load_f.write(json.dumps(all_answer_tail, indent=1))