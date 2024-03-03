import argparse
import json
import logging
import os
import re
import time
import openai
from tqdm import tqdm 
import multiprocessing as mp
from prompt_selection import Demon_sampler

class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0
    
    def get_response(self, input_text, turn_type):
        if self.args.debug:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            response = message['content'].strip()
        return response


    
    def create_message(self, input_text, turn_type):
        if turn_type == "init_query":  
            instruction = self.prompt['init_query']
            input_text = instruction

        elif turn_type == "first_give_demonstration":
            template = self.prompt['first_give_demonstration']
            question = input_text
            input_text = template.format(question=question)
        elif turn_type == "analogy_demonstration":
            template = self.prompt['analogy_demonstration']
            analogy_demons = input_text
            input_text = template.format(selected_analogy_demonstrations=analogy_demons)
        elif turn_type == "supplement_demonstration":
            template = self.prompt['supplement_demonstration']
            supplement_demons = input_text
            input_text = template.format(selected_supplement_demonstrations=supplement_demons)
        elif turn_type == "candidate_score":
            template = self.prompt['candidate_score']
            question = input_text
            input_text = template.format(question = question)
        elif turn_type == "directly_ask":  
            template = self.prompt['directly_ask']
            question,candidate = input_text
            input_text = template.format(question=question, order_of_candidate=candidate)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                if args.debug_online:
                    print(res)
                self.token_num = res['usage']['total_tokens']
                return res['choices'][0]['message']
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(30)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)


    def reset_history(self):
        self.history_messages = []
        self.history_contents = []
        self.token_num = 0
        
    def reset_history_messages(self):
        self.history_messages = []

    def reset_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]
        
        
from collections import defaultdict

import tiktoken

class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)
        self.max_llm_input_token = args.max_llm_input_tokens
        self.prompt_selector = Demon_sampler(args)
        
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []
        
        self.id2ent = defaultdict(str)
        self.ent2id = defaultdict(str)
        self.rel2id= defaultdict(str)
        self.text2ent = defaultdict(str)
        self.ent2text = defaultdict(str)
        self.all_candidate_answers = defaultdict(list)
        self.rel2text_align = defaultdict(str)
        self.rel2text = defaultdict(str)
        self.scores = []
        
        self.load_rel_txt_to_id()
        self.load_ent_map_id()
        self.load_all_candidate_answers()
        self.load_ent_to_text()
        self.load_relation_text()
        
    def count_token(self, string):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
        return len(encoding.encode(string))
    
                
    def forward(self, question, tpe): #Here tpe_id not a int id, but like '/m/08966'
        self.LLM.reset_history()
        self.reset_history()
        final_order = []
        tpe_str = self.ent2text[tpe]
        candidate_ids = self.all_candidate_answers['\t'.join([str(self.ent2id[tpe]),str(self.rel2id[question])])]
        for id in candidate_ids[:args.candidate_num]:
            self.candidate_answers.append(self.ent2text[self.id2ent[str(id)]])
        
        if args.query == 'tail':
            question_text = self.generate_demonstration_text((tpe_str, question,''))
        elif args.query == 'head':
            question_text = self.generate_demonstration_text(('', question, tpe_str))
    
        init_response = self.LLM.get_response((''), "init_query")
        assert self.check_work_flow(init_response),"LLM Not Understand Task"
        
        second_response = self.LLM.get_response(question_text, "first_give_demonstration")
        
        effective_demon_step = 0
        # current_demon_step = -1
        current_demon_step = 0

        # true_candidates = self.prompt_selector.true_candidates(tpe, question)
        while effective_demon_step < args.eff_demon_step and current_demon_step < args.max_demon_step:
            
            analogy_demons, supplement_demons = self.prompt_selector.randomsampler(tpe,question,args.demon_per_step,current_demon_step)
            analogy_demon_text = self.serialize_demonstrations(analogy_demons)
            supplement_demon_text = self.serialize_demonstrations(supplement_demons)
            if analogy_demon_text == "None." and supplement_demon_text == "None.": break

            if analogy_demon_text != "None":
                current_demon_response = self.LLM.get_response((analogy_demon_text),"analogy_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens: 
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    break
            if supplement_demon_text != "None.":
                current_demon_response = self.LLM.get_response((supplement_demon_text),"supplement_demonstration")
                if self.LLM.token_num >= args.max_llm_input_tokens: 
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    break
            current_demon_step += 1
            
            score_dict = defaultdict(int)
        for can in self.candidate_answers:
            triplet = (tpe_str, question, can) if args.query == 'tail' else (can, question, tpe_str)
            can_text = self.generate_demonstration_text(triplet)
            can_response = self.LLM.get_response((can_text),"candidate_score")
            score = self.parse_result(can_response, "candidate_score")
            can_text_short = can.split(':')[0].strip()
            score_dict[can_text_short] = score

            self.LLM.history_contents.pop()
            self.LLM.history_messages.pop()
            final_order.append(score)
        
            
        return score_dict, self.LLM.history_contents, self.log

        
        
    
    def check_work_flow(self, response):
        if "no" in response.lower():
            return False
        return True
        
        
    def relation_text(self,relation):
        if self.args.align_text:
            return self.rel2text_align[relation]
        else:
            return self.rel2text[relation]
    

    def serialize_demonstrations(self,demon_triples):
        demon_text = ""
        for tp in demon_triples:
            demon_text += self.generate_demonstration_text(tp) + '\n'
        demon_text.strip()
        if demon_text == "": demon_text = "None."
        return demon_text
    
    def generate_demonstration_text(self, triple):
        h,r,t = triple
        demonstration_text = ""
        if self.args.query == 'tail':
            if t != "":
                if self.args.align_text:
                    demonstration_text += self.relation_text(r).replace("[H]", h).replace("[T]", t) + ". "
                else:
                    demonstration_text += t + " " + self.relation_text(r) + " " + h +". "
            else:
                demonstration_text += self.relation_text(r).replace("[H]", h).replace("[T]", "Which word") + "? "
        elif self.args.query == 'head':
            if h != "":
                if self.args.align_text:
                    demonstration_text += self.relation_text(r).replace("[H]", h).replace("[T]", t) + ". "
                else:
                    demonstration_text += t + " " + self.relation_text(r) + " " + h +". "
            else:
                demonstration_text += self.relation_text(r).replace("[T]", t).replace("[H]", "Which word") + "? "
        
        return demonstration_text
        
    def parse_result(self, response, parse_type):
        response = response.lower()
        if parse_type == "final_answer":
            if "the final order:" in response:
                final_order = re.split(":",response)[1].strip().strip('.').strip('\[').strip('\]')
            return final_order
        elif parse_type == "candidate_score":
            score = response
            return score
    
    def reset_history(self):
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []
        self.scores = []
        
    def load_all_candidate_answers(self):
        with open("dataset/" + self.args.dataset + "/retriever_candidate_"+ args.query +".txt",'r') as load_f:
            self.all_candidate_answers=json.load(load_f)
            
    def load_relation_text(self):
        with open("dataset/" + self.args.dataset + "/alignment/alignment_clean.txt",'r') as load_f:
            self.rel2text_align=json.load(load_f)  
        with open("dataset/" + self.args.dataset + "/relation2text.txt",'r') as load_f:
            self.rel2text=json.load(load_f)      
            
    def load_rel_txt_to_id(self):
        with open('dataset/' + self.args.dataset + '/get_neighbor/relation2id.txt', 'r') as file:
            relation_lines = file.readlines()
            for line in relation_lines:
                _name, _id = line.strip().split("\t")
                self.rel2id[_name] = _id
                
                
    def load_ent_map_id(self):
        with open('dataset/' + self.args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                _name, _id = line.strip().split("\t")
                self.ent2id[_name] = _id
                self.id2ent[_id] = _name
    
    
    def load_ent_to_text(self):
        with open('dataset/' + self.args.dataset + '/entity2text.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text
                self.text2ent[text] = ent

                
        
                   
                    
    
def main(args, all_data, idx, api_key):
    from collections import defaultdict
    

    import openai
    openai.api_key = api_key
    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), output_path))
    solver = Solver(args)

    count = 0
    valid_count = 0
    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data)):
                count += 1
                try:
                    tpe = sample['HeadEntity'] if args.query == 'tail' else sample['Answer']
                    question = sample['Question']
                    
                    prediction, chat_history, record = solver.forward(question, tpe)
                    valid_count += 1
                except openai.error.InvalidRequestError as e:
                    print(e)
                    continue
                except Exception as e:
                    logging.exception(e)
                    continue

                chat = str(sample["ID"]) + "\n" + "\n******\n".join(chat_history) + "\nAnswers: " + str(
                    sample['Answer']) + "\n------------------------------------------\n"
                fclog.write(chat)

                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")

    print("---------------PID %d end with %d/%d samples--------------" % (os.getpid(), valid_count, count))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wn18rr")
    parser.add_argument('--candidate_num', default=50, type=int)
    parser.add_argument('--output_path', default="./outputs/wn18rr/output_score.txt")
    parser.add_argument('--chat_log_path', default="./outputs/wn18rr/chat_score.txt")
    parser.add_argument('--query', default="tail", required=True)
    parser.add_argument('--model_path', default=None)
    
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_online', action="store_true")

    parser.add_argument('--align_text', action="store_true")
    
    parser.add_argument('--max_tokens', default=300, type=int, help='max-token')
    parser.add_argument('--prompt_path', default="./prompts/link_prediction_scoring.json")
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--bagging_type', default="llm", )
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--device', default=0, help='the gpu device')
    
    parser.add_argument('--api_key', default="", type=str)
    parser.add_argument('--demon_per_step', default=8)
    parser.add_argument('--eff_demon_step', default=4)
    parser.add_argument('--max_demon_step', default=4)
    parser.add_argument('--max_llm_input_tokens', default=3750, type=int)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')


    args = parser.parse_args()
    args.output_path = './outputs/'+ args.dataset +'/output_'+ args.query +'_scoring.txt'
    args.chat_log_path = './outputs/'+ args.dataset +'/chat_'+ args.query +'_scoring.txt'

    print("Start querying the LLM.")
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
    test_triplet = []


    with open("dataset/" + args.dataset + "/test_answer.txt",'r') as load_f:
        test_triplet=json.load(load_f)
    print("Totally %d test examples." % len(test_triplet))



    if args.debug_online:
        test_triplet = test_triplet[0:2*args.num_process]
    if args.num_process == 1:
        main(args, test_triplet, idx=-1, api_key=args.api_key)
    else:
        num_each_split = int(len(test_triplet) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(test_triplet))
            else:
                end = (idx + 1) * num_each_split
            split_data = test_triplet[start:end]
            try:
                p.apply_async(main, args=(args, split_data, idx, all_keys[idx]))
            except Exception as e:
                logging.exception(e)

        p.close()
        p.join()
        print("All of the child processes over!")
        
#   python3 link_prediction_scoring.py --dataset wn18rr --debug_online --query head
#   python3 link_prediction_scoring.py --dataset wn18rr --debug_online --query head