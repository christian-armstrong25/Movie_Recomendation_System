import numpy as np
import torch
import random

# read header of mat_comp to get k,n,m
file = open('mat_comp','r')
n, m, k = map(int, file.readline().split())

# fill M with data from mat_comp
M, M_validation = np.zeros((n, m)), np.zeros((n, m))
validation_split = 0.10

for a in range(k):
  f = file.readline().split()
  i = int(f[0])-1 # adjust to 0 based indexing
  j = int(f[1])-1
  M_i_j = float(f[2])

  # randomized the position in the modulo that you sample from
  rand = random.randint(0,1//validation_split - 1)
  if a % (1//validation_split) == 0: # update index of next random sample
    rand = random.randint(0,1//validation_split - 1)

  if a % (1//validation_split) == rand: # put in validation set
    M_validation[i,j] = M_i_j
  else: # put in training set
    M[i,j] = M_i_j

# hyperparameters
r = 24 # rank
s = 0.2 # step size
T = 150 # iterations
LAMBDA = 0.2 # l2 regularization coef

# SVD to get X and Y
print("SVD...")
U, S, Vh = np.linalg.svd(M)
X = np.matmul(U[:,:r], np.diag(S[:r])) # combine U and S for X
Y = Vh.T[:,:r]
Omega, Omega_validation = np.copy(M), np.copy(M_validation)
Omega[Omega != 0] = 1
Omega_validation[Omega_validation != 0] = 1

# Turn everything into Tensors
X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
Y_T = torch.tensor(Y.T, requires_grad=True, dtype=torch.float32)
M = torch.tensor(M, dtype=torch.float32)
M_validation = torch.tensor(M_validation, dtype=torch.float32)
Omega = torch.tensor(Omega, dtype=torch.float32)
Omega_validation = torch.tensor(Omega_validation, dtype=torch.float32)

# Matrix Completion
optimizer = torch.optim.Adagrad([X,Y_T], lr=s)
A = [] 
for t in range(T):
    # Compute the current approximation and loss
    A = torch.matmul(X, Y_T)
    l = torch.sum(torch.pow((M-A)*Omega,2)) / torch.sum(Omega)
    # regularizer
    l += LAMBDA * (torch.mean(torch.pow(X,2)) + torch.mean(torch.pow(Y_T,2)))
    
    # optimize
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    # cross validation
    l_validation = torch.sum(torch.pow((M_validation-A)*Omega_validation,2)) / torch.sum(Omega_validation)
    print(f"epoch {t}, validation loss: {l_validation}")

# get final loss from cross validation
A = A.detach().numpy() # switch to nparray
A = np.clip(A, 0.75, 4.75) # clip values outside of range
A = torch.tensor(A, dtype=torch.float32)
l_validation = torch.sum(torch.pow((M_validation-A)*Omega_validation,2)) / torch.sum(Omega_validation)
print(f"FINAL LOSS: {l_validation}")

# get q
q = int(file.readline())

# output predictions for q queries
output_file = open('mat_comp_ans','w')
A = A.detach().numpy()

for a in range(q):
  f = file.readline().split()
  i = int(f[0])-1 # adjust to 0 based indexing
  j = int(f[1])-1

  output = str(A[i,j])
  if a != q-1: # no newline on last line
    output += "\n"
    
  output_file.write(output)