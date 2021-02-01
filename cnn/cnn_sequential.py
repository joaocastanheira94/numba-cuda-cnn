import time
from . import cnn
import numpy as np
import utils.utils as u

def forward(inputs, labels, c_s, p_s, conv_w, conv_b, fully_w, fully_b, pool_size):
    start_conv = time.time()
    ## --------CONV FEED FORWARD -----------------
    conv_output_shape = cnn.calculate_convolution_output_dims(inputs.shape, conv_w.shape, c_s)
    n, h_in, w_in, _ = inputs.shape
    _, h_out, w_out, _ = conv_output_shape
    h_f, w_f, _, n_f = conv_w.shape
    conv_output = np.zeros(conv_output_shape)
    #convolution process: apply filters to batch data
    #pass input data through weigths (filters)
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * c_s
            h_end = h_start + h_f
            w_start = j * c_s
            w_end = w_start + w_f
            conv_output[:, i, j, :] = np.sum(
                inputs[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                conv_w[np.newaxis, :, :, :],
                axis=(1, 2, 3)
            )

    conv_output = conv_output + conv_b

    ## ---- RELU activation function -----##
    conv_output = np.maximum(0,conv_output)

    pooling_cache = {}
    n, h_in, w_in, c = conv_output.shape
    h_pool, w_pool = pool_size
    h_out = 1 + (h_in - h_pool) // p_s
    w_out = 1 + (w_in - w_pool) // p_s
    max_pool_output = np.zeros((n, h_out, w_out, c))

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * p_s
            h_end = h_start + h_pool
            w_start = j * p_s
            w_end = w_start + w_pool
            a_prev_slice = conv_output[:, h_start:h_end, w_start:w_end, :]
            max_pool_output[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))
            #save mask (needed in backwards)
            mask = np.zeros_like(a_prev_slice)
            n, h, w, c = a_prev_slice.shape
            a_prev_slice = a_prev_slice.reshape(n, h * w, c)
            idx = np.argmax(a_prev_slice, axis=1)

            n_idx, c_idx = np.indices((n, c))
            mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
            pooling_cache[(i,j)] = mask
        
    end_conv = time.time()-start_conv
    start_fully = time.time()
    ##---- FLATTEN LAYER ----####
    #more info in: https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-3-flattening
    flatten = np.ravel(max_pool_output).reshape(max_pool_output.shape[0],-1)

    ##----- FULLY CONNECTED LAYER (MLP) (CLASSIFICATION LAYER) ----
    mlp_output = np.dot(flatten, fully_w) + fully_b
        
    #
    #softmax
    e = np.exp(mlp_output - mlp_output.max(axis=1, keepdims=True))
    preds = e / np.sum(e, axis=1, keepdims=True)
    end_fully = time.time()-start_fully
    return preds, flatten, end_conv, end_fully,pooling_cache

def backward(inputs,labels, preds, fully_w, fully_b,conv_w, conv_b, flatten, max_pool_output_shape, conv_output_shape,pooling_cache, c_s, p_s,pool_size):    
    start = time.time()
    ##backward
    for i in range(preds.shape[0]):
        preds[i, labels[i]] -= 1
    
    grad = preds

    ##fully connected backward
    g_inputs = np.dot(grad, fully_w.T)
    g_weights = np.zeros(shape=fully_w.shape, dtype=np.float32)
    g_biases = np.zeros(shape=fully_b.shape, dtype=np.float32)
    for i in range(grad.shape[0]):
        g_weights += (np.dot(flatten[i][:, np.newaxis], grad[i][np.newaxis, :]))
        g_biases += grad[i]
    grad = g_inputs

    ##flatten backward
    grad = grad.reshape(max_pool_output_shape)

       ##pooling backward
    g_max_pool_output = np.zeros(shape=conv_output_shape)
    _, h_out, w_out, _ = grad.shape
    h_pool, w_pool = pool_size
    
    for i in range(h_out):
        for j in range(w_out):
            h_start = i * p_s
            h_end = h_start + pool_size[1]
            w_start = j * p_s
            w_end = w_start + pool_size[0]
            g_max_pool_output[:, h_start:h_end, w_start:w_end, :] += \
                grad[:, i:i + 1, j:j + 1, :] * pooling_cache[(i, j)]
                     
    g_max_pool_output2 = g_max_pool_output.copy()
    grad = g_max_pool_output

    ##conv backward
    _, h_out, w_out, _ = grad.shape
    n, h_in, w_in, _ = inputs.shape
    h_f, w_f, _, _ = conv_w.shape

    cv_g_biases = grad.sum(axis=(0, 1, 2)) / n
    cv_g_weights = np.zeros_like(conv_w)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * c_s
            h_end = h_start + h_f
            w_start = j * c_s
            w_end = w_start + w_f
            cv_g_weights += np.sum(
                inputs[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                grad[:, i:i+1, j:j+1, np.newaxis, :],
                axis=0
            )
    return cv_g_weights, cv_g_biases, g_weights, g_biases, time.time()-start
    

def train_one_epoch(X, y, c_s, p_s,conv_w, conv_b, fully_w, fully_b,pool_size):
    #forward
    preds, flatten, duration_conv, duration_fully, pooling_cache = forward(X, y, c_s, p_s,conv_w, conv_b, fully_w, fully_b,pool_size)
    predictions = preds.copy()
    conv_output_shape = cnn.calculate_convolution_output_dims(X.shape, conv_w.shape, c_s)
    max_pool_output_shape = cnn.calculate_maxpool_output_dims(conv_output_shape, p_s, pool_size)
    #backward
    cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases, duration_backward = backward(X, y, preds, fully_w, fully_b,conv_w, conv_b,flatten,max_pool_output_shape,conv_output_shape,pooling_cache, c_s,p_s,pool_size)
    return predictions, cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases


def train_sequential(data, batch_size,kernel_shape, filters, c_s, p_s, pool_size, num_classes,lr,epochs):
    begin = time.time()
    
    train_data, train_label, test_data, test_label = data

    #init network
    input_dims = (batch_size, train_data.shape[1],train_data.shape[2],train_data.shape[3])
    conv_w, conv_b, fully_w, fully_b = cnn.init_network(input_dims,kernel_shape, filters,c_s, p_s, pool_size,num_classes)
    conv_output_shape = (cnn.calculate_convolution_output_dims(input_dims, conv_w.shape,c_s))
    max_pool_output_shape = (cnn.calculate_maxpool_output_dims(conv_output_shape, p_s, pool_size))

    evaluate_test_set = 10
    
    epochs_time = []
    accs = []
    
    for e in range(epochs):
        epoch_begin = time.time()
        X, y = u.next_batch(train_data, train_label, batch_size)

        preds, cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases \
            = train_one_epoch(X, y,c_s, p_s, conv_w, conv_b, fully_w, fully_b,pool_size)
        
        conv_w, conv_b, fully_w, fully_b = cnn.update_parameters(conv_w, conv_b, fully_w, fully_b, cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases,lr, batch_size)
        acc = cnn.softmax_accuracy(y, preds)
        accs.append(acc)
        epoch_duration = time.time()-epoch_begin
        epochs_time.append(epoch_duration)
        acc = cnn.softmax_accuracy(y, preds)
        accs.append(acc)
        
    duration = time.time()-begin
    model = (conv_w,conv_b,fully_w,fully_b)
    return duration, epochs_time, accs, conv_w.size + conv_b.size + fully_w.size + fully_b.size,model
