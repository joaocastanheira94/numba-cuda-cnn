# Training a CNN in parallel

This project was developed in the context of the Master's degree in Data Science at Faculty of Sciences, University of Lisbon.

We implemented the training of a simple Convolution Neural Network in parallel, by using CPU multiprocessing capabilities and GPU programming, through CUDA.

For more details regarding our implementation, check [report](report.pdf)

## CNN architecture

The network architecture is very simple: a convolution and max pooling layer, followed by a fully connected layer for classification. Softmax was used as error function.


## Results

We compared the time execution of the training, for three different versions: the sequential version, which was developed in NumPy, based on [1], and both the parallel version on CPU and GPU.


![Results](images/parameters_comparison.png?raw=true "Title")


## Code structure

The project code is structured in the following way:
    
    project.ipynb: In this notebook, different CNNs with different settings are trained in sequential, parallel on CPU and parallel on GPU ways. The results from this experiment were saved in df_results pickle file.
    results.ipynb: The results from the experience above described are analysed.
    
    folder cnn: all the code necessary to train the CNN in multiple ways: sequential, parallel on CPU and parallel on GPU
        
        -> cnn.py: code for network initialization, and some helper functions;
        -> cnn_sequential: code for training sequentially on CPU by using NumPy;
        -> cnn_parallel_cpu.py: code for training in multithreading CPU;
        -> cnn_parallel_cuda.py: code for training in GPU. It includes all the cuda functions/kernels developed.
    
    folder tests:
    
        -> conv_filter_example.ipynb: a simple example of the convolution process
        -> test_sequential_vs_gpu.ipynb: a notebook containg code for one epoch of training, in CUDA GPU and CPU sequential version. There you can see that the output of the training process is exactly the same for both versions, for the same randomly initialized weights.
        
## References

[1]https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net

[2] https://github.com/WHDY/mnist_cnn_numba_cuda
        