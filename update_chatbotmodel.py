# -*- coding: utf-8 -*-
import numpy as np
import pickle
import json
from jieba import lcut
from tqdm import tqdm
def get_array(ta):
    temp = []
    err = []
    for i in lcut(ta.replace(' ', '')):
        if i in words:
            temp.append(word_to_index[i])
        else:
            err.append(i)
            words.append(i)
            w = i
            idx = max(word_to_index.values())+1
            word_to_index[w] = idx
            index_to_word[idx] = w
            temp.append(word_to_index[i])
    temp_o = temp.copy()
    if len(temp) <20:
        times = 20- len(temp)
        temp_o.extend([0]*times)
    temp = temp[:20]
    temp_o = temp_o[:20]
    return temp, temp_o


main_path = 'C://Users//user//Desktop//project//ChatBot_threebotbyJimmy//data_chatbot//'
question = np.load(main_path+'pad_question.npy', allow_pickle=True)
answer = np.load(main_path+'pad_answer.npy', allow_pickle=True)
answer_o = np.load(main_path+'answer_o.npy', allow_pickle=True)
with open(main_path+ 'vocab_bag.pkl', 'rb') as f:
    words = pickle.load(f)
with open(main_path+'pad_word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open(main_path+ 'pad_index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

    
vocab_size = len(word_to_index) + 1
maxLen=20

with open(main_path+ 'pad_index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

file = open("D://datasource//baike_qa_valid.json", 'r', encoding='utf-8')
papers = []
for line in file.readlines():
    dic = json.loads(line)
    papers.append(dic)

answer_o_v2 = [i for i in answer_o]
for i in tqdm(range(len(papers))):
    qq = papers[i]['title']
    aa = papers[i]['answer']    

    q_temp_, q_temp_o = get_array(qq)
    a_temp_, a_temp_o = get_array(aa)

    answer_o_v2.append(a_temp_)
    answer_v2 = np.concatenate((answer, np.array(a_temp_o).reshape(1,20)), 0)
    question_v2= np.concatenate((question, np.array(q_temp_o).reshape(1,20)), 0)    
   
answer_o_v2 = np.array(answer_o_v2)  
#save json and load json file
np.save(main_path+'pad_question_v2.npy', question_v2)
np.save(main_path+'pad_answer_v2.npy', answer_v2)
np.save(main_path+'answer_o_v2.npy', answer_o_v2)
with open(main_path+ 'vocab_bag_v2.pkl', 'wb') as fp:
    pickle.dump(words, fp)    

with open(main_path+ 'pad_word_to_index_v2.pkl', 'wb') as fp:
    pickle.dump(word_to_index, fp)        

with open(main_path+ 'pad_index_to_word_v2.pkl', 'wb') as fp:
    pickle.dump(index_to_word, fp)