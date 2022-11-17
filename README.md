# Compressed_SVD
TO DO:  
  1.Build Dataset: Mnist and Cifar10  
  *Implement MLP with Numpy  
  *Implement Truncated SVD algorithm Numpy  
  *Implement K-SVD with Numpy  
  *Train MLP  
  *Compress MLP with Truncated SVD(Try different K), recover each layer: W=W_k  
  *Compress MLP with Truncated SVD(Try different K), recover each layer: W_1=U_k, W_2=V_k (one layer becomes two)  
  *Set a thresh hold: lambda, range over the weight in each layer, weight=0(where abs(weight)<lambda), repeat the process  
  *Compress MLP with K-SVD, to be explored  
  *Compare the accuracy,running time and storage with each of the compression methods.  
