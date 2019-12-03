# --------------
# Code starts here

import numpy as np

# Code starts here

# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

# Compute eigenvalues and eigencevectrs
eigenvalues, eigenvectors = np.linalg.eig(adj_mat)

# Eigen vector corresponding to 1

eigen_1 = abs(eigenvectors[:,0])
# normalzing the vector

eigen_1 = eigen_1/np.linalg.norm(eigenvectors[:,0],1)

# most important page , finding maximum value within the eigenvector
print(eigen_1)
page = np.argmax(eigen_1) + 1

print(page)

# Code ends here


# --------------
# Code starts here

# Initialize stationary vector I
init_I = np.array([1,0,0,0,0,0,0,0])

print(init_I.shape)
print(adj_mat.shape)

# Perform iterations for power method
for i in range(10):
    init_I = np.dot(adj_mat,init_I)/np.linalg.norm(init_I,1) 
    

power_value = np.max(init_I)

power_page = np.argmax(init_I) + 1 # np.where(np.max(adj_mat) == power_value)

print(power_value)
print(power_page)


# Code ends here


# --------------
# Code starts here

# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

# Initialize stationary vector I
new_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(10):
    #getting the dot product for adjacency matrix and stationary vector , also normalising the result
    new_init_I = np.dot(new_adj_mat,new_init_I)/np.linalg.norm(new_init_I,1)

print(new_init_I)
# Here we get pagerank value for 3rd webpage as zero. Is it not possible as it has incoming connections.
# Code ends here


# --------------
# Alpha value
alpha = 0.85

# Code starts here

# Modified adjancency matrix(new hyperlink matrix)
#Using G=αS+(1−α)n1 ,1 as the new hyperlink matrix and 1 is a n*n matrix whose all entries are 1.α can #take any value between 0 and 1.
G = alpha * new_adj_mat + (1 - alpha)*(1/len(new_adj_mat))*np.ones(new_adj_mat.shape)

# Initialize stationary vector I
final_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
# taking dot product of adjacency matrix and stationary vector and normalising the same
for i in range(1000):
    final_init_I = np.dot(G,final_init_I)/np.linalg.norm(final_init_I,1)

print(final_init_I)

# Code ends here


