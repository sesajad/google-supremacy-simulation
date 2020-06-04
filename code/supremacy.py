import numpy as np
from random import choice

n = 5
cycles = 6
state_shape = (2, 2, 2, 2, 2)

# define single-qubit gates
xsqrt = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
ysqrt = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
wsqrt = np.array([[1, -np.sqrt(1j)], [np.sqrt(-1j), 1]]) / np.sqrt(2)

single_gates = [xsqrt, ysqrt, wsqrt]

# define double-qubit gates
double_gate = np.array([[1, 0, 0, 0], \
                        [0, 0, -1j, 0], \
                        [0, -1j, 0, 0], \
                        [0, 0, 0, np.exp(1j * np.pi/6)]])
double_gate = np.reshape(double_gate, (2, 2, 2, 2))

# double-qubit gate pattern (from shape) ABCDCDAB
pattern = [(1,2), (2,3), (0,2), (2,4), (0,2), (2,4), (1,2), (2,3)]

# sample from different 100 random circuits
samples_U = 100

# list of F values
fs = []

for _ in range(samples_U):
  last_applied_gate = [None] * n

  # define input state
  state = np.reshape(np.array([1] + 31 * [0]), state_shape)

  # iterate over cycles
  for c in range(cycles):
    # iterate over qubits to apply single gates 
    for i in range(n):
      # apply a random gate on ith qubit
      gate = choice([g for g in single_gates if np.all(g != last_applied_gate[i])])
      state = np.tensordot(gate, state, axes=(1, i))
      
    # apply double-qubit gate
    state = np.tensordot(double_gate, state, axes=((2,3), pattern[c % len(pattern)]))
    
  # last half cycle
  for i in range(n):
    gate = choice([g for g in single_gates if np.all(g != last_applied_gate[i])])
    state = np.tensordot(gate, state, axes=(1, i))
    
  # let's dice!
  ps = (abs(state)**2).flatten()
  # number of samples from output of this circuit
  samples_q = 10
  for _ in range(samples_q):
    fs.append(2**n * np.random.choice(ps, p=ps) - 1)
  
print('F = ', np.mean(fs), '±', np.std(fs) / np.sqrt(len(fs) - 1))
