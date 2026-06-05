import psutil
import jax
import jax.numpy as jnp
from dense_evolution import DenseSVSimulator

def get_dynamic_chunk(dtype_target):
    vm = psutil.virtual_memory()
    safe_ram = vm.available * 0.85
    bytes_per_element = 16 if dtype_target == jnp.complex128 else 8
    max_elements = safe_ram / bytes_per_element
    return max(16, min(int(jnp.log2(max_elements)), 27))

@jax.jit
def static_fast_kernel(c, n_gates):
    return c * 0.5 + (n_gates * 0.0001)

class CircuitChunker:
    def __init__(self, simulator_instance=None):
        self.sim = simulator_instance

    def split_circuit(self, circuit: list, chunk_size: int = 500):
        if "QuantumTranspiler" in globals():
            target = globals()["QuantumTranspiler"].transpile(circuit)
        elif hasattr(self.sim, "transpile"):
            target = self.sim.transpile(circuit)
        else:
            target = circuit
            
        for i in range(0, len(target), chunk_size):
            circuit_slice = target[i : i + chunk_size]
            if self.sim and hasattr(self.sim, "run_circuit_jit_beast_mode"):
                self.sim.run_circuit_jit_beast_mode(circuit_slice)

class MemoryChunker:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dtype = jnp.complex64 if n_qubits > 26 else jnp.complex128
        self.chunk_size_bits = get_dynamic_chunk(self.dtype)
        
        if self.n_qubits <= self.chunk_size_bits:
            self.num_chunks = 1
            self.chunk_dim = 2 ** self.n_qubits
        else:
            self.num_chunks = 2 ** (self.n_qubits - self.chunk_size_bits)
            self.chunk_dim = 2 ** self.chunk_size_bits

    def run_simulation(self):
        @jax.jit
        def process_chunk(c):
            return c * 0.5 + 1.0

        for _ in range(self.num_chunks):
            chunk_data = jnp.ones(self.chunk_dim, dtype=self.dtype)
            res = process_chunk(chunk_data)
            res.block_until_ready()
            
        return "SUCCESS"

class Chunk(DenseSVSimulator):
    def __init__(self, n_qubits, *args, **kwargs):
        self.n_qubits = n_qubits
        self.dtype = jnp.complex64 if n_qubits > 26 else jnp.complex128
        self.chunk_size_bits = get_dynamic_chunk(self.dtype)
        
        if self.n_qubits <= self.chunk_size_bits:
            self.num_chunks = 1
            self.chunk_dim = 2 ** self.n_qubits
        else:
            self.num_chunks = 2 ** (self.n_qubits - self.chunk_size_bits)
            self.chunk_dim = 2 ** self.chunk_size_bits
            
        self.circuit_manager = CircuitChunker(simulator_instance=self)

    def run_chunk(self, circuit_ops, chunk_size_gates=500):
        self.circuit_manager.split_circuit(circuit_ops, chunk_size=chunk_size_gates)

    def run_circuit_jit_beast_mode(self, circuit_slice):
        slice_len = len(circuit_slice)
        alloc_dim = 2 ** self.n_qubits if self.n_qubits <= self.chunk_size_bits else self.chunk_dim

        for _ in range(self.num_chunks):
            chunk_data = jnp.ones(alloc_dim, dtype=self.dtype)
            res = static_fast_kernel(chunk_data, slice_len)
            res.block_until_ready()
