import numpy as np
import jax.tree_util as jtu

class PyTreeReplayBuffer:
    """
    A generic experience replay buffer capable of storing and sampling data of arbitrary PyTree structures.
    For example, it can store transitions like (GraphsTuple, action, reward, ...).
    """
    def __init__(self, capacity: int, dummy_input):
        """
        Initialize the Buffer.
        
        Args:
            capacity: Maximum capacity of the buffer (number of transitions).
            dummy_input: A "dummy" or "template" object with the exact same PyTree structure 
                         as a single data sample to be stored. For example, a transition tuple.
        """
        self.capacity = int(capacity)
        self.edge_capacity = int(2e5)  # configurable
        self.ptr = 0
        self.edge_pointer = 0
        self.size = 0
        self.edge_size = 0
        
        # --- Core initialization steps ---
        # 1. Flatten the dummy input to get the list of leaf nodes and the PyTree structure definition
        flat_input, self.tree_def = jtu.tree_flatten(dummy_input)
        
        # 2. Create a corresponding NumPy storage array for each flattened leaf node
        #    The first dimension of each storage array is the capacity.
        self.buffers = [
            np.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype)
            for leaf in flat_input
        ]
        self.edge_buffers = [
            np.zeros((self.capacity, *leaf.shape), dtype=leaf.dtype)
            for leaf in flat_input
        ]
        
    def add_batch(self, batch_data):
        """
        Add a batch of data to the buffer.

        Args:
            batch_data: A batched PyTree, whose leaf nodes all have a leading 
                        batch dimension. For example, (100, ...).
        """
        # 1. Determine the amount of data to add this time
        flat_batch_data = jtu.tree_leaves(batch_data)
        num_to_add = flat_batch_data[0].shape[0]
        
        # 2. Calculate the index range to write to
        #    This logic handles the wrap-around case when the buffer is full
        if self.ptr + num_to_add <= self.capacity:
            # a. If there is enough space, write directly
            idxs = np.arange(self.ptr, self.ptr + num_to_add)
            
            for i, leaf_batch in enumerate(flat_batch_data):
                self.buffers[i][idxs] = leaf_batch
        else:
            # b. If not enough space, write in two parts (circular buffer)
            #    Part 1: fill to the end
            num_part1 = self.capacity - self.ptr
            idxs_part1 = np.arange(self.ptr, self.capacity)
            
            #    Part 2: start writing from the beginning
            num_part2 = num_to_add - num_part1
            idxs_part2 = np.arange(0, num_part2)

            flat_batch_data = jtu.tree_leaves(batch_data)
            for i, leaf_batch in enumerate(flat_batch_data):
                # Write part 1
                self.buffers[i][idxs_part1] = leaf_batch[:num_part1]
                # Write part 2
                self.buffers[i][idxs_part2] = leaf_batch[num_part1:]
        
        # 3. Update pointer and size
        self.ptr = (self.ptr + num_to_add) % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)
    
    def add_edge(self, edge_N, episode_transitions):
        """
        Called at the end of an episode to process the entire trajectory and store it in the buffer.
        is_edge_fn: A function that takes state s as input and returns a boolean (whether it is in the edge region).
        """
        if edge_N != -1:
            episode_edge_transitions = jtu.tree_map(
                lambda x: x[0:edge_N],
                episode_transitions
            )
            flat_batch_data = jtu.tree_leaves(episode_edge_transitions)
            num_to_add = flat_batch_data[0].shape[0]
        
            # 2. Calculate the index range to write to
            #    This logic handles the wrap-around case when the buffer is full
            if self.edge_pointer + num_to_add <= self.edge_capacity:
                # a. If there is enough space, write directly
                idxs = np.arange(self.edge_pointer, self.edge_pointer + num_to_add)
                for i, leaf_batch in enumerate(flat_batch_data):
                    self.edge_buffers[i][idxs] = leaf_batch
            else:
                # b. If not enough space, write in two parts (circular buffer)
                #    Part 1: fill to the end
                num_part1 = self.edge_capacity - self.edge_pointer
                idxs_part1 = np.arange(self.edge_pointer, self.edge_capacity)
                
                #    Part 2: start writing from the beginning
                num_part2 = num_to_add - num_part1
                idxs_part2 = np.arange(0, num_part2)

                for i, leaf_batch in enumerate(flat_batch_data):
                    # Write part 1
                    self.edge_buffers[i][idxs_part1] = leaf_batch[:num_part1]
                    # Write part 2
                    self.edge_buffers[i][idxs_part2] = leaf_batch[num_part1:]
            
            # 3. Update pointer and size
            self.edge_pointer = (self.edge_pointer + num_to_add) % self.edge_capacity
            self.edge_size = min(self.edge_size + num_to_add, self.edge_capacity)
            
    def add(self, data):
        """
        Add a single data sample to the buffer (e.g., a transition).
        
        Args:
            data: A PyTree object with the exact same structure as the dummy_input during initialization.
        """
        # 1. Flatten the input data into a list of leaf nodes
        flat_data = jtu.tree_leaves(data)
        
        # 2. Iterate through the leaf list and store each leaf in its corresponding NumPy storage array
        for i, leaf in enumerate(flat_data):
            self.buffers[i][self.ptr] = leaf
            
        # 3. Update pointer and current size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of data from the buffer.
        
        Args:
            batch_size: Batch size.
        
        Returns:
            A batched PyTree with the same structure as a single sample, but all leaf nodes
            have an additional leading batch dimension.
        """
        # 1. Generate random indices
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # 2. Slice data from each NumPy storage array based on the indices
        sampled_leaves = [buf[idxs] for buf in self.buffers]
        
        # 3. Reconstruct the sampled leaves into a PyTree using the structure definition saved during initialization
        main_batch = jtu.tree_unflatten(self.tree_def, sampled_leaves)
        
        edge_batch = None
        if self.edge_size > batch_size:
            edge_idx = np.random.randint(0, self.edge_size, size=batch_size)
            sampled_leaves = [buf[edge_idx] for buf in self.edge_buffers]
            edge_batch = jtu.tree_unflatten(self.tree_def, sampled_leaves)
        
        return main_batch, edge_batch
        
    def __len__(self):
        return self.size