import numpy as np
from threading import Thread
import multiprocessing
from time import time
from . import cnn 
from . import cnn_sequential
import random
import utils.utils as u

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

#from: https://stackoverflow.com/a/6894023/6444477
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def merge_results(deltas,num_threads, conv_w, conv_b, fully_w, fully_b):
    cv_g_weights2, cv_g_biases2 , fully_g_weigths2, fully_g_biases2 = np.zeros_like(conv_w), np.zeros_like(conv_b), np.zeros_like(fully_w), np.zeros_like(fully_b)
    for i in range(num_threads):
        cv_g_weightsi, cv_g_biasesi , fully_g_weigthsi, fully_g_biasesi = deltas[i]
        cv_g_weights2+= cv_g_weightsi
        cv_g_biases2+=cv_g_biasesi
        fully_g_weigths2+=fully_g_weigthsi
        fully_g_biases2+=fully_g_biasesi
    return cv_g_weights2, cv_g_biases2 , fully_g_weigths2, fully_g_biases2

def train_parallel(data, batch_size,kernel_shape, filters, c_s, p_s, pool_size, num_classes, num_threads,lr,epochs):
    begin = time()
    
    train_data, train_label, test_data, test_label = data
    
    #init network
    input_dims = (batch_size, train_data.shape[1],train_data.shape[2],train_data.shape[3])
    conv_w, conv_b, fully_w, fully_b = cnn.init_network(input_dims,kernel_shape, filters,c_s, p_s, pool_size,num_classes)
    conv_output_shape = (cnn.calculate_convolution_output_dims(input_dims, conv_w.shape,c_s))
    max_pool_output_shape = (cnn.calculate_maxpool_output_dims(conv_output_shape, p_s, pool_size))

    split_size = batch_size // num_threads     

    evaluate_test_set = 10
    epochs_time = []
    accs = []
    for e in range(epochs):
        threads = [] 
        threads_results = []
        epoch_begin = time()
        X, y = u.next_batch(train_data, train_label, batch_size)

        #split work across threads
        for i in range(num_threads):
            start = i * split_size
            end = None if i+1 == num_threads else (i+1) * split_size 
      
            X_t = X[start:end]
            y_t = y[start:end]

            t = ThreadWithReturnValue(target=cnn_sequential.train_one_epoch, args=(X_t, y_t, c_s, p_s ,conv_w, conv_b, fully_w, fully_b,pool_size))
            
            threads.append(t)         
            threads[-1].start() 

        # wait for all threads to finish                                            
        for t in threads:                                                           
            r = t.join()     
            threads_results.append(r)

        threads_results = np.array(threads_results)
        preds = np.vstack(threads_results[:,0])
        deltas = threads_results[:,1:]

        #merge threads deltas
        cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases = merge_results(deltas,num_threads,conv_w, conv_b, fully_w, fully_b)

        #update parameters
        conv_w, conv_b, fully_w, fully_b = cnn.update_parameters(conv_w, conv_b, fully_w, fully_b, cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases,lr,batch_size)
       
        acc = cnn.softmax_accuracy(y, preds)
        accs.append(acc)
        epoch_duration = time()-epoch_begin
        epochs_time.append(epoch_duration)

    duration = time()-begin
    model = (conv_w,conv_b,fully_w,fully_b)
    return duration, epochs_time, accs, conv_w.size + conv_b.size + fully_w.size + fully_b.size,model