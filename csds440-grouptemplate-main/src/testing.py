# List of indices to exclude
masked_indices = [0, 2]

# List of information gains
infogains = [0.5, 0.3, 0.4, 0.5]

# Create a new list of (index, gain) tuples, excluding the masked indices
masked_infogains = [(i, gain) for i, gain in enumerate(infogains) if i not in masked_indices]

# Find the tuple with the maximum gain in the masked list
max_gain_tuple = max(masked_infogains, key=lambda x: x[1])

# The first element of this tuple is the index in the original list
max_gain_index = max_gain_tuple[0]
max_gain_value = max_gain_tuple[1]

print(f"Maximum information gain (unmasked) is {max_gain_value} at index {max_gain_index} in the original list.")
