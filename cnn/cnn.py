import numpy as np

def calculate_convolution_output_dims(input_dims, conv_w_dims, stride):
    n, h_in, w_in, _ = input_dims
    h_f, w_f, _, n_f = conv_w_dims
    h_out = (h_in - h_f) // stride + 1
    w_out = (w_in - w_f) // stride + 1
    return n, h_out, w_out, n_f

def calculate_maxpool_output_dims(conv_output_dims, stride, pool_size):
    n, h_in, w_in, c = conv_output_dims
    h_pool, w_pool = pool_size
    h_out = 1 + (h_in - h_pool) // stride
    w_out = 1 + (w_in - w_pool) // stride
    return n, h_out, w_out, c

def init_network(input_dims, kernel_shape, filters, c_s,p_s, pool_size, num_classes):
    conv_w = np.random.randn(*kernel_shape, filters) * 0.1
    conv_b = np.random.randn(filters) * 0.1
    
    conv_output_dims = (calculate_convolution_output_dims(input_dims, conv_w.shape, c_s))
    maxpool_output_dims = (calculate_maxpool_output_dims(conv_output_dims,p_s, pool_size))
    
    max_pool_output = np.zeros(shape=maxpool_output_dims)
    flatten = np.ravel(max_pool_output).reshape(max_pool_output.shape[0],-1)
    
    fully_w = np.random.randn(flatten.shape[1],num_classes) * 0.1
    fully_b = np.random.randn(1, num_classes) * 0.1
    
    return conv_w, conv_b, fully_w, fully_b

def softmax_accuracy(y, probs):
    y_pred = np.argmax(probs, axis=1)
    one_hot_matrix = np.zeros_like(probs)
    one_hot_matrix[np.arange(probs.shape[0]), y_pred] = 1
    #one hot encode vector
    y_ohe = np.zeros((y.size, y.max()+1))
    y_ohe[np.arange(y.size),y] = 1
    return (one_hot_matrix == y_ohe).all(axis=1).mean()

def update_parameters(conv_w, conv_b, fully_w, fully_b, cv_g_weights, cv_g_biases , fully_g_weigths, fully_g_biases,LR,BATCH_SIZE):
    conv_w -= cv_g_weights * LR / BATCH_SIZE
    conv_b -= cv_g_biases * LR / BATCH_SIZE
    fully_w -= fully_g_weigths * LR / BATCH_SIZE
    fully_b -= fully_g_biases * LR / BATCH_SIZE
    return conv_w, conv_b, fully_w, fully_b
