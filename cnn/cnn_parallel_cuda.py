import numba 
import math
from numba import cuda, float32
import time
from . import cnn 
from . import cnn_sequential
import utils.utils as u
import numpy as np

def get_thread_ct(batch_size):
    if batch_size <= 250:
        return 32
    elif batch_size <= 500:
        return 64
    elif batch_size <=1000:
        return 128
    elif batch_size <=2000:
        return 256
    else: return 1024
    

def train_cuda(data, batch_size,kernel_shape, filters, c_s, p_s, pool_size, num_classes,lr,epochs):
    begin = time.time()
    
    train_data, train_label, test_data, test_label = data

    #init network
    input_dims = (batch_size, train_data.shape[1],train_data.shape[2],train_data.shape[3])
    conv_w, conv_b, fully_w, fully_b = cnn.init_network(input_dims,kernel_shape, filters,c_s, p_s, pool_size,num_classes)
    conv_output_shape = (cnn.calculate_convolution_output_dims(input_dims, conv_w.shape,c_s))
    max_pool_output_shape = (cnn.calculate_maxpool_output_dims(conv_output_shape, p_s, pool_size))
    
    epochs_time = []
    dt_time = []

    accs = []
    
    for e in range(epochs):
        epoch_begin = time.time()
        
        X, y = u.next_batch(train_data, train_label, batch_size)

        #INIT INTERMEDIARY MATRICES (NEEDED DURING TRAINING)
        conv_output = np.zeros(shape=conv_output_shape, dtype=np.float32)
        max_pool_output = np.zeros(shape=max_pool_output_shape, dtype=np.float32)
        gradients = np.zeros(shape=(batch_size, num_classes), dtype=np.float32)
        preds = np.zeros(shape=gradients.shape, dtype = np.float32)
        mlp_output = np.zeros(shape=gradients.shape, dtype=np.float32)

        g_flatten = np.zeros(shape=(batch_size, max_pool_output_shape[1]*max_pool_output_shape[2]*max_pool_output_shape[3]), dtype=np.float32)
        g_max_pool_output = np.zeros(shape=conv_output_shape, dtype=np.float32)

        #INIT WEIGHTS DELTAS
        g_fully_w = np.zeros(shape=fully_w.shape, dtype=np.float32)
        g_fully_b = np.zeros(shape=fully_b.shape, dtype=np.float32)
        g_conv_w = np.zeros(shape=(batch_size,conv_w.shape[0], conv_w.shape[1],conv_w.shape[2],conv_w.shape[3]))
        g_conv_b = np.zeros(shape=(batch_size, conv_b.shape[0]), dtype=np.float32)
        
        begin_dt = time.time()

        # TRANSFER DATA TO GPU: TRAININ DATA, WEIGHTS, INTERMEDIARY MATRICES AND DELTAS
        
        #train data and labels
        Xg = cuda.to_device(X)
        yg = cuda.to_device(y)

        conv_outputg = cuda.to_device(conv_output)
        max_pool_outputg = cuda.to_device(max_pool_output)
        predsg = cuda.to_device(preds)
        gradientsg = cuda.to_device(gradients)
        mlp_outputg = cuda.to_device(mlp_output)

        g_max_pool_outputg = cuda.to_device(g_max_pool_output)
        g_flatten_g = cuda.to_device(g_flatten)

        g_fully_wg = cuda.to_device(g_fully_w)
        g_fully_bg = cuda.to_device(g_fully_b)
        g_conv_wg = cuda.to_device(g_conv_w)
        g_conv_bg = cuda.to_device(g_conv_b)

        conv_wg = cuda.to_device(conv_w)
        conv_bg = cuda.to_device(conv_b)
        fully_wg = cuda.to_device(fully_w)
        fully_wg_t = cuda.to_device(fully_w.T)
        fully_bg = cuda.to_device(fully_b)

        duration_to_device = time.time()-begin_dt

        #conv & max pooling forwarding
        thread_ct = get_thread_ct(batch_size)
        
        block_ct = (math.ceil(batch_size/thread_ct),1)
        conv_pool_forward[block_ct, thread_ct](Xg,conv_wg,conv_bg,conv_outputg,max_pool_outputg)

        #copy maxpooloutput from  gpu back to the cpu, to reshape it. 
        #We tried to do that with cuda data object, but it throwed an error.

        flatten = max_pool_outputg.copy_to_host()
        flatten = flatten.ravel().reshape(batch_size, -1)
        flatteng = cuda.to_device(flatten)
        
        #fc layer forward
        thread_ct = (32, 32)
        block_ct = list(map(lambda x: int(math.ceil(float(x) / thread_ct[0])), [gradientsg.shape[0], gradientsg.shape[1]]))
        fully_gpu_forward[block_ct, thread_ct](flatteng, fully_wg, fully_bg, mlp_outputg,gradientsg,predsg, yg)

        #fc layer backward
        thread_ct = (32, 32)
        block_ct = list(map(lambda x: int(math.ceil(float(x) / thread_ct[0])), [flatteng.shape[0], flatteng.shape[1]]))
        fully_gpu_backward[block_ct, thread_ct](flatteng,gradientsg,fully_wg_t,g_fully_wg,g_fully_bg, g_flatten_g)
        
        
        #update fc layer deltas
        block_ct = list(map(lambda x: int(math.ceil(float(x) / thread_ct[0])), [fully_w.shape[0], fully_w.shape[1]]))
        fully_gpu_gradients[block_ct, thread_ct](flatteng,gradientsg,fully_wg_t,g_fully_wg,g_fully_bg, g_flatten_g)
        
        
        #conv pool backwards & deltas update
        g_flatten_g = g_flatten_g.reshape(max_pool_output_shape)
        thread_ct = get_thread_ct(batch_size)
        block_ct = list(map(lambda x: int(math.ceil(float(x) / thread_ct)), [batch_size, 1]))
        conv_pool_backward[block_ct, thread_ct](Xg, conv_outputg, g_flatten_g, g_max_pool_outputg, g_conv_wg,g_conv_bg)

        X = Xg.copy_to_host()
        y = yg.copy_to_host()

        begin_dt = time.time()
        
        #conv_output = conv_outputg.copy_to_host()
        #max_pool_output = max_pool_outputg.copy_to_host()
        #flatten = flatteng.copy_to_host()
        #gradients = gradientsg.copy_to_host()
        #mlp_output = mlp_outputg.copy_to_host()
#
        preds = predsg.copy_to_host()
##
        #g_max_pool_output = g_max_pool_outputg.copy_to_host()
        #g_flatten = g_flatten_g.copy_to_host()
#
        conv_w = conv_wg.copy_to_host()
        conv_b = conv_bg.copy_to_host()
        fully_w = fully_wg.copy_to_host()
        fully_b = fully_bg.copy_to_host()
        fully_w_t = fully_wg_t.copy_to_host()
        
        g_fully_w = g_fully_wg.copy_to_host()
        g_fully_b = g_fully_bg.copy_to_host()
        g_conv_w = g_conv_wg.copy_to_host()
        g_conv_b = g_conv_bg.copy_to_host()
        
        duration_dt = time.time()-begin_dt

        duration_data_transfer = time.time()-begin_dt + duration_to_device
        duration = time.time()-begin
        
        #sum gradients output gpu
        g_conv_w = np.sum(g_conv_w, axis=0)
        g_conv_b = np.sum(g_conv_b, axis=0)/batch_size
        
        #update parameters
        conv_w, conv_b, fully_w, fully_b = cnn.update_parameters(conv_w, conv_b, fully_w, fully_b, g_conv_w, g_conv_b , g_fully_w, g_fully_b,lr, batch_size)
        acc = cnn.softmax_accuracy(y, preds)
        
        accs.append(acc)
        epoch_duration = time.time()-epoch_begin
        epochs_time.append(epoch_duration)
        dt_time.append(duration_data_transfer)

    duration = time.time()-begin
    model = (conv_w,conv_b,fully_w,fully_b)
    
    return duration, epochs_time,dt_time, accs, conv_w.size + conv_b.size + fully_w.size + fully_b.size,model

#####CUDA FUNCTIONS###########

#CONVOLUTION AND POOLING FORWARDING => DATA PARALLELISM
@cuda.jit(debug=False)
def conv_pool_forward(imgs, conv_w,conv_b ,conv_output,max_pool_output):
    x = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    index = bx * bdx + x
      
    conv_stride = 1
    pool_stride = 2
    pool_size = (2,2)
    numThreads = bdx * cuda.gridDim.x

    if index<imgs.shape[0]:
        img = imgs[index]
        #filter
        for f in range(conv_w.shape[3]):
           #output pixel i
           for i in range(conv_output.shape[1]):
               #output pixel j 
               for j in range(conv_output.shape[2]):
                    h_start = i * conv_stride
                    h_end = h_start + conv_w.shape[0]
                    w_start = j * conv_stride
                    w_end = w_start + conv_w.shape[1]
                    
                   # #multiply image blocks with kernels
                    ih = 0
                    for h in range(h_start,h_end):
                        iw = 0
                        for w in range(w_start,w_end):
                            conv_output[index,i, j, f] += img[h,w,0]*conv_w[ih,iw,0,f]
                            iw += 1
                        ih += 1        

                    #add bias
                    conv_output[index, i, j, f] += conv_b[f]

                    #relu activation function
                    conv_output[index, i, j, f] = max(0,conv_output[index, i, j, f])
        z = 0
        for f in range(max_pool_output.shape[3]):
            for i in range(max_pool_output.shape[1]):
                for j in range(max_pool_output.shape[2]):
                    h_start = i * pool_stride
                    h_end = h_start + pool_size[0]
                    w_start = j * pool_stride
                    w_end = w_start + pool_size[1]
                    
                    for h in range(h_start,h_end):
                        for w in range(w_start,w_end):
                            maximum = max(max_pool_output[index, i, j, f], conv_output[index, h, w, f])
                            max_pool_output[index, i, j, f] = maximum
                    z += 1
                    
#FULLY CONNECTED LAYER FORWARD
@cuda.jit
def fully_gpu_forward(flatten, fully_w, fully_b, mlp_output, output,preds, y):
    i, j = cuda.grid(2)
 
    if i < output.shape[0] and j < output.shape[1]:
        tmp = 0
        for k in range(flatten.shape[1]):
            tmp += flatten[i, k] * fully_w[k, j]

        output[i, j] =  tmp + fully_b[0, j]
        mlp_output[i, j] =  tmp + fully_b[0, j]
        
        cuda.syncthreads()
        
        maximum = 0
        for z in range(output.shape[1]):
            maximum = max(maximum, output[i,z])

        #softmax
        output[i,j] = math.exp(output[i,j]-maximum)
        cuda.syncthreads()

        #prediction
        sum =0
        for z in range(output.shape[1]):
            sum += output[i,z]
        pred = output[i,j]/sum

        output[i,j] = pred
        preds[i,j] = pred
        
        cuda.syncthreads()

        #softmax backward
        if j == y[i]:
            output[i,j] -= 1

        
        cuda.syncthreads()
        
#FC LAYER BACKWARD
@cuda.jit()
def fully_gpu_backward(flatten, gradient, fully_w_t, g_fully_w, g_fully_b, g_flatten):
    i, j = cuda.grid(2) 
    if i < g_flatten.shape[0] and j < g_flatten.shape[1]
                
        tmp = 0
        for z in range(gradient.shape[1]):
            tmp+=gradient[i, z] * fully_w_t[z,j]
        g_flatten[i,j] = tmp
        
        
#UPDATE FC LAYER GRADIENTS
@cuda.jit()
def fully_gpu_gradients(flatten, gradient, fully_w_t, g_fully_w, g_fully_b, g_flatten):
    i, j = cuda.grid(2)

    if i < g_fully_w.shape[0] and j < g_fully_w.shape[1]:
        tmp = 0
        for k in range(flatten.shape[0]):
            g = gradient[k, j]
            tmp += g
            g_fully_w[i, j] += flatten[k,i] * g
        g_fully_b[0,j] = tmp

#CONV & MAXPOOLING BACKWARD
@cuda.jit(debug=False)
def conv_pool_backward(X2, conv_output, g_flatten, g_max_pool_output, cv_g_weights, cv_g_biases):
    index = cuda.grid(1)
    conv_stride = 1
    pool_stride = 2
    pool_size=(2,2)
    if index<conv_output.shape[0]:
        a = conv_output[index]
        inpu = X2[index]
        for f in range(g_flatten.shape[3]):

            for i in range(g_flatten.shape[1]):
                for j in range(g_flatten.shape[2]):

                    h_start = i * pool_stride
                    h_end = h_start + pool_size[0]
                    w_start = j * pool_stride
                    w_end = w_start + pool_size[1]
                    
                    #max and argmax
                    m = -1000000
                    ih_, iw_ = 0,0
                    for h in range(h_start,h_end):
                        for w in range(w_start,w_end):
                            v = a[h, w, f] 
                            if v > m: 
                                m = v
                                ih_ = h
                                iw_ = w
                    
                    for h in range(h_start,h_end):
                        for w in range(w_start,w_end):
                            if (h == ih_ and w == iw_):
                                g_max_pool_output[index, h, w,f] += g_flatten[index, i, j, f]
            

            for i in range(conv_output.shape[1]):
                for j in range(conv_output.shape[2]):

                    h_start = i * conv_stride
                    h_end = h_start + cv_g_weights.shape[1]
                    w_start = j * conv_stride
                    w_end = w_start + cv_g_weights.shape[2]
                    
                    cv_g_biases[index, f] =  cv_g_biases[index, f] + g_max_pool_output[index, i, j, f] 
 
                    ih = 0
                    for h in range(h_start,h_end):
                        iw = 0
                        for w in range(w_start,w_end):
                            cv_g_weights[index, ih, iw, 0, f] += inpu[h,w,0] * g_max_pool_output[index, i, j, f]

                            iw +=1
                        ih += 1
        cuda.syncthreads()
