# Compressed_SVD

## Dependencies

```
python >= 3.7
tqdm
tabulate
numpy
```

## Setting up Conda environment

```sh
conda create --name mlfoundie python=3.10
conda activate mlfoundie
conda install -c conda-forge scipy sympy numpy pandas ipython ipywidgets ipykernel jupyterlab jupytext matplotlib "nbconvert[webpdf]"
```

## TO DO

- [x]Build Dataset: Mnist and Cifar10  
- [x]Implement MLP with Numpy  
- [x]Implement L2 norm in MLP with Numpy
- [x]Implement Truncated SVD algorithm Numpy    
- [x]Train MLP  
- [ ]Compress MLP with Truncated SVD(Try different K), recover each layer: W=W_k  
- [ ]Compress MLP with Truncated SVD(Try different K), recover each layer: W_1=U_k, W_2=V_k (one layer becomes two)  
- [ ]Set a thresh hold: lambda, range over the weight in each layer, weight=0(where abs(weight)<lambda), repeat the process  d  
- [ ]Compare the accuracy,running time and storage with each of the compression methods.  
