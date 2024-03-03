'''
Prompt Engineering: 1. Add comparable  2. Add confidence scoreto better and confident to be best. 3. Final Have a try again 4. rethinking 5.vote 6.repeat
7.iterative or accumulated update  8.question is/are
'''
import json
from gensim.summarization.bm25 import BM25
from gensim import corpora
import operator
from collections import defaultdict
import random
import argparse
import heapq
class Demon_sampler:
    def __init__(self, args):
        self.ent2text = defaultdict(str)
        self.entity_supplement = defaultdict(list)
        self.relation_analogy = defaultdict(list)
        self.T_link_base = defaultdict(list)
        self.link_base = defaultdict(list)
        self.link_base_txt = defaultdict(list)
        self.args = args
        self.dataset = args.dataset
        self.load_ent_to_text()
        self.load_demonstration()
        self.shrink_link_base()
        self.demo_list_execution = []
        
    def load_demonstration(self):
        with open("dataset/" + self.dataset + "/demonstration/"+ self.args.query +"_supplement.txt", "r") as f:
            supplement_pool = json.load(f)

        with open("dataset/" + self.dataset + "/demonstration/"+ self.args.query +"_analogy.txt", "r") as f:
            analogy_pool = json.load(f)

        keys = self.ent2text.keys()
        for key in supplement_pool:
            tmp_list = []
            for value in supplement_pool[key]:
                if value[0] in keys:
                    tmp_list.append([self.ent2text[value[0]],value[1],self.ent2text[value[2]]])
                else:
                    tmp_list.append([value[0],value[1],value[2]])

            self.entity_supplement[key] = tmp_list
        for key in analogy_pool:
            tmp_list = []
            for value in analogy_pool[key]:
                if value[0] in keys:
                    tmp_list.append([self.ent2text[value[0]],value[1],self.ent2text[value[2]]])
                else:
                    tmp_list.append([value[0],value[1],value[2]])
                # random.shuffle(tmp_list)
            self.relation_analogy[key] = tmp_list
        for key in self.link_base:
            tmp_list = []
            for value in self.link_base[key]:
                if value[0] in keys:
                    tmp_list.append([self.ent2text[value[0]],value[1],self.ent2text[value[2]]])
                else:
                    tmp_list.append([value[0],value[1],value[2]])
                
            self.link_base_txt[key] = tmp_list

            
    def true_candidates(self,h,r):
        return self.T_link_base['\t'.join([h,r])][2]
          
    def Diversity_arranged(self, tpe, relation):
        demon_list = self.relation_analogy['\t'.join([tpe, relation])]
        entity_counter = defaultdict(int)
        def count_sum(triple):
            return entity_counter[triple[0]] + entity_counter[triple[2]], triple
        priority_queue = [count_sum(triple) for triple in demon_list]
        heapq.heapify(priority_queue)

        sorted_list = []
        while priority_queue:
            _, next_triple = heapq.heappop(priority_queue)
            sorted_list.append(next_triple)
            entity_counter[next_triple[0]] += 1
            entity_counter[next_triple[2]] += 1
            priority_queue = [count_sum(triple) for triple in priority_queue]
            heapq.heapify(priority_queue)
        self.relation_analogy['\t'.join([tpe, relation])] = sorted_list

        
    def BM25_arranged(self, tpe, relation):
        demon_list = self.entity_supplement['\t'.join([tpe, relation])]
        tpe_text = self.ent2text[tpe]
        question_text = tpe_text + relation if self.args.query == 'tail' else relation + tpe_text
        texts = ['\t'.join(triple) for triple in demon_list]
        dictionary = corpora.Dictionary([text.split() for text in texts])
        corpus = [dictionary.doc2bow(text.split()) for text in texts]
        bm25 = BM25(corpus)
        query = dictionary.doc2bow(question_text.split())
        scores = bm25.get_scores(query)
        scored_triples = list(zip(demon_list, scores))
        sorted_triples = sorted(scored_triples, key=lambda x: x[1], reverse=True)
        sorted_demon_list = [triple for triple, score in sorted_triples]
        self.entity_supplement['\t'.join([tpe, relation])] = sorted_demon_list

        
    def poolsampler(self, tpe, r, num, step_num):
        analogy_num = num//2
        supplement_num = num - analogy_num
        start_analogy = step_num * analogy_num
        end_analogy = start_analogy + analogy_num
        start_supple = step_num * supplement_num
        end_supple = start_supple + supplement_num
        if '\t'.join([tpe, r]) not in self.demo_list_execution:
            self.Diversity_arranged(tpe, r)
            self.BM25_arranged(tpe, r)
            self.demo_list_execution.append('\t'.join([tpe, r]))
        analogy_arranged_set = self.relation_analogy['\t'.join([tpe, r])]
        supplement_arranged_set = self.entity_supplement['\t'.join([tpe, r])]
        analogy_set = analogy_arranged_set[start_analogy:end_analogy]
        supplement_set = supplement_arranged_set[start_supple:end_supple]
        return analogy_set,supplement_set
          
    def randomsampler(self, tpe, r, num, step_num): # need a new version for no repeat facts
        analogy_num = num//2
        supplement_num = num - analogy_num
        start_analogy = step_num * analogy_num
        end_analogy = start_analogy + analogy_num
        start_supple = step_num * supplement_num
        end_supple = start_supple + supplement_num
        analogy_set = self.relation_analogy['\t'.join([tpe, r])][start_analogy:end_analogy]
        supplement_set = self.entity_supplement['\t'.join([tpe, r])][start_supple:end_supple]
        return analogy_set,supplement_set

        
    def shrink_link_base(self):
        with open("dataset/" + self.dataset + "/demonstration/"+ "T_link_base_"+ self.args.query +".txt", "r") as f:
            self.link_base = json.load(f)
        
        for key in self.link_base:
            if len(self.link_base[key]) == 0: 
                self.T_link_base[key] = []
                break
            h,r = key.split('\t')
            enetity_link_base = ""
            for value in self.link_base[key][:10]:
                h_text = self.ent2text[value[0]]
                enetity_link_base += self.ent2text[value[2]] + ','
            enetity_link_base.strip(',')
            # if enetity_link_base == "": enetity_link_base = "None"
            self.T_link_base[key] = [h_text,r,enetity_link_base]

    def load_ent_to_text(self):
            with open('dataset/' + self.dataset + '/entity2text.txt', 'r') as file:
                entity_lines = file.readlines()
                for line in entity_lines:
                    ent, text = line.strip().split("\t")
                    self.ent2text[ent] = text
                  
    def true_candidate_v2(self,h,r,num):
          true_set = self.link_base_txt['\t'.join([h,r])][:num]
          return true_set
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default=None)
#     args = parser.parse_args()
#     data_sampler = Demon_sampler(args)
#     maxlen = 0
#     maxkey = ''
#     for key in data_sampler.T_link_base:
#         print(len(data_sampler.T_link_base[key][2]))
#         if len(data_sampler.T_link_base[key][2]) > maxlen:
#             maxlen = len(data_sampler.T_link_base[key][2])
#             maxkey = key
#     print(maxlen,maxkey)