import struct
import random
import sys

def write_bin_file(filename, num_vertices, edges):
    """
    Writes a graph to a .bin file compatible with sg0/tric.

    Args:
        filename (str): The output .bin file name.
        num_vertices (int): The total number of vertices (M_).
        edges (list of tuple): A list of (source, target, weight) edges.
                               Vertices IDs should be 0-indexed.
    """
    # 1. Prepare the data structures
    num_edges = len(edges)

    # Build the CSR index array
    # Initialize an array of size (num_vertices + 1) with zeros
    index = [0] * (num_vertices + 1)
    # Count the degree of each vertex
    for src, tgt, w in edges:
        index[src + 1] += 1
    # Create a cumulative sum to get the start indices
    for i in range(1, num_vertices + 1):
        index[i] += index[i-1]

    # Flatten the edge list into a list of (target, weight) pairs
    # First, create a temporary array of lists to store neighbors
    adj_list = [[] for _ in range(num_vertices)]
    for src, tgt, w in edges:
        adj_list[src].append((tgt, w))
    # Then, flatten it in the order of the CSR index
    flat_edges = []
    for i in range(num_vertices):
        flat_edges.extend(adj_list[i])

    # 2. Write to the binary file
    with open(filename, 'wb') as f:
        # Write header (2 unsigned long longs)
        f.write(struct.pack('QQ', num_vertices, num_edges))

        # Write vertex index array (num_vertices + 1 unsigned long longs)
        for val in index:
            f.write(struct.pack('Q', val))

        # Write edge list (for each edge, write target and weight)
        for tgt, w in flat_edges:
            f.write(struct.pack('Qd', tgt, w))  # 'Q' for unsigned long long, 'd' for double

# --- Example Usage ---
if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print(f"Usage: ./{sys.argv[0]} <num_vertices> <num_edges> <output_file>")
        exit(1)
    else:
        V = int(sys.argv[1])
        E = int(sys.argv[2])
        output_file = sys.argv[3]

        # Generate random edges (0-indexed vertices)
        random_edges = []
        for _ in range(E):
            src = random.randint(0, V-1)
            tgt = random.randint(0, V-1)
            if src != tgt:  # Avoid self-loops
                weight = 1.0
                random_edges.append((src, tgt, weight))

        # Remove potential duplicates to keep the graph simple
        random_edges = list(set(random_edges))
        print(f"Generated {len(random_edges)} unique random edges.")

        # Write to a .bin file
        write_bin_file(output_file, V, random_edges)
        print(f"Graph written to {output_file}")
