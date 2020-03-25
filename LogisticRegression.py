import os
import numpy as np

def get_train_test_data(train_dir,test_dir) :
    train_data = {}
    test_data = {}

    train_data['spam'] = os.listdir(train_dir+'spam')
    train_data['ham'] = os.listdir(train_dir+'ham')

    test_data['spam'] = os.listdir(test_dir+'spam')
    test_data['ham'] = os.listdir(test_dir+'ham')

    return train_data,test_data

def get_vocabulary(train_data,test_data,train_dir,test_dir):
    vocab = []
    for class_val,data_file in list(train_data.items()):
        for file_name in data_file:
            with open(train_dir+class_val+'/'+file_name,'r',encoding='utf-8',errors='ignore') as file_data:
                token = file_data.read().strip().split()
                vocab += set(token)
    for class_val,data_file in list(test_data.items()):
        for file_name in data_file:
            with open(test_dir+class_val+'/'+file_name,'r',encoding='utf-8',errors='ignore') as file_data:
                token = file_data.read().strip().split()
                vocab += set(token)
    return set(vocab)

def get_word_freq(train_data, train_dir,vocab, train_len):
    train_X = np.zeros((train_len,len(vocab)))
    train_Y = np.zeros(train_len)
    i = 0
    for class_val, data_file in list(train_data.items()):
        for file_name in data_file:
            with open(train_dir + class_val + '/' + file_name, 'r', errors='ignore', encoding='utf-8') as file_data:
                words = file_data.read().strip().split()
                words = set(words)
                for st in words:
                    index = vocab.index(st)
                    train_X[i][index] = train_X[i][index] + 1
            if class_val == 'spam':
                train_Y[i] = 1;
            i = i+1
    return train_X,train_Y


def log_regression(train_data,train_dir,vocab,learning_rate):
    train_len = len(train_data['spam']) + len(train_data['ham'])
    train_X, train_Y = get_word_freq(train_data, train_dir, vocab, train_len)
    regul_parameter = 0.00001
    bias = np.full((train_X.shape[0], 1), 0.5)
    W = np.full((train_X.shape[1],1),0.5)
    for i in range(1000):
        z = np.dot(train_X, W)
        z = z.astype('float128')
        z = z + bias
        pred_Y = 1 / (1 + np.exp(-z))
        diff = pred_Y - train_Y
        reg_term = regul_parameter / train_X.shape[0] * np.transpose(W)
        derivative_term = ((np.dot(diff, train_X)) / train_X.shape[0]) - reg_term
        new_W = W - (np.transpose(learning_rate * derivative_term))
        W = new_W
    return W


def get_accuracy(test_data,test_dir,trained_W, vocab):
    test_len = len(test_data['spam']) + len(test_data['ham'])
    test_X, test_Y = get_word_freq(test_data, test_dir, vocab, test_len)
    crct_count = 0
    total_count = 0
    for i in range(test_len):
        x_value = test_X[i,:]
        z = np.dot(x_value,trained_W)
        z = z.astype('float128')
        pred_Y = 1 / (1 + np.exp(-z))
        if np.any(pred_Y) > 0.5:
            predicted_class = 1
        else :
            predicted_class = 0

        if(predicted_class == test_Y[i]):
            crct_count = crct_count + 1
        total_count = total_count + 1
    accuracy = (float(crct_count / total_count)) * float(100)
    return accuracy

if __name__ == '__main__' :
    D_train_dir = "data/train/"
    D_test_dir = "data/test/"
    D_dir = "data/"
    D_train_data, D_test_data = get_train_test_data(D_train_dir, D_test_dir)
    D_vocab = list(get_vocabulary(D_train_data,D_test_data,D_train_dir,D_test_dir))

    learning_rate = [0.1,0.01,0.001,0.0001]

    for l in learning_rate:
        trained_W = log_regression(D_train_data,D_train_dir,D_vocab,l)
        D_accuracy = get_accuracy(D_test_data,D_test_dir,trained_W,D_vocab)
        print("Data Accuracy without stop words for learning rate ",l," is ", D_accuracy, "%")