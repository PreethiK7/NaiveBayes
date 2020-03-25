import os
import math

def get_train_test_data(train_dir,test_dir) :
    train_data = {}
    test_data = {}

    train_data['spam'] = os.listdir(train_dir+'spam')
    train_data['ham'] = os.listdir(train_dir+'ham')

    test_data['spam'] = os.listdir(test_dir+'spam')
    test_data['ham'] = os.listdir(test_dir+'ham')

    return train_data,test_data

def get_vocabulary(train_data,train_dir) :
    vocabs = []
    for class_val,data_file in list(train_data.items()):
        for file_name in data_file:
            with open(train_dir+class_val+'/'+file_name,'r',errors='ignore',encoding='utf-8') as file_data:
                token = file_data.read().strip().split()
                vocabs += set(token)
    return set(vocabs)

def count_doc(train_data) :
    count = 0
    for files in list(train_data.values()):
        count = count + len(files)
    return count

def get_word_freq(file_dir,file_list,train_dir):
    word_freq = {}
    word_count = 0
    for file_name in file_list:
        with open(train_dir+file_dir+'/'+file_name,'r',errors='ignore',encoding='utf-8') as file_data:
            token = file_data.read().strip().split()
            words = set(token)
            word_count = word_count + len(words)
            for st in words:
                if st in word_freq:
                    word_freq[st] = word_freq[st]+1
                else:
                    word_freq[st] = 1
    return word_freq,word_count

def get_cond_prob(words_freq,words_count,vocab_len):
    prob = {}
    for class_val,words in list(words_freq.items()):
        prob[class_val] = {}
        prob[class_val]['N/A'] = float(1) / (words_count[class_val] + vocab_len + 1)
        for w in words:
            prob[class_val][w] = float(words[w] + 1) / (words_count[class_val] + vocab_len + 1)
    return prob


def train_data(train_data,train_dir) :
    vocab = get_vocabulary(train_data,train_dir)
    n = count_doc(train_data)
    priors = {}
    word_freq = {}
    word_count = {}
    for class_val,data_val in list(train_data.items()):
        priors[class_val] = len(data_val) / float(n)
        word_freq[class_val],word_count[class_val] = get_word_freq(class_val,data_val,train_dir)
    cond_prob = get_cond_prob(word_freq,word_count,len(vocab))
    return vocab,priors,cond_prob



def classify_data(data,priors,cond_prob):
    max_likelihood = float('-inf')
    class_value = ''
    words = set(data)
    for class_val,prior_val in list(priors.items()):
       cur_likelihood = float(math.log(prior_val))
       for w in words:
           if w in cond_prob[class_val]:
               cur_likelihood = cur_likelihood + math.log(cond_prob[class_val][w])
           else:
               cur_likelihood = cur_likelihood + math.log(cond_prob[class_val]['N/A'])
       if cur_likelihood>max_likelihood:
            max_likelihood = cur_likelihood
            class_value = class_val
    return class_value

def get_accuracy(test_data,test_dir,priors,cond_prob):
    total_count = 0
    total_crct_count = 0

    for real_class,file_list in list(test_data.items()):
        class_crct_count = 0
        for file_name in file_list:
            with open(test_dir+real_class+'/'+file_name,'r',encoding='utf-8',errors='ignore') as file_data:
                text_data = file_data.read().strip().split()
                pred_class = classify_data(text_data,priors,cond_prob)
                if pred_class == real_class:
                    class_crct_count = class_crct_count + 1
                total_count = total_count + 1
        print("For class ",real_class," correctly predicted count(with stop words) is ",class_crct_count," for a total of ",total_count)
        total_crct_count = total_crct_count + class_crct_count
    accuracy = (float(total_crct_count)/total_count) * float(100)
    return accuracy


if __name__ == '__main__' :
    D_train_dir = "data/train/"
    D_test_dir = "data/test/"
    D_dir = "data/"
    D_train_data, D_test_data = get_train_test_data(D_train_dir,D_test_dir)

    D_vocab, D_prior, D_cond_prob = train_data(D_train_data,D_train_dir)
    D_accuracy = get_accuracy(D_test_data,D_test_dir, D_prior, D_cond_prob)
    print("Data Accuracy with stop words is ", D_accuracy ,"%")