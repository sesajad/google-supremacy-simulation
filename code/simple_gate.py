import numpy

state_shape = (2, 2, 2, 2, 2)
state = np.reshape(np.array([1] + 31 * [0]), state_shape)

# a valid quantum state must have norm = 1
assert(np.linalg.norm(state) == 1.0)

gate = np.matrix([[0, 1j], [-1j, 0]])

# a valid quantum gate must be unitary (maintains norm)
assert(np.all(np.matmul(gate.H, gate) == np.identity(2)))

# apply gate on 3th qubit (2nd if we start from 0) 
np.tensordot(gate, state, axes=(1, 2))

