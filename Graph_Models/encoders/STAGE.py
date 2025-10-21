"""
TODO:
model needs to accommodate for the following
1. variable number of nodes in input graphs (node number can change)
2. edge attributes (not just connectivity)
3. spatial attention per sequence (which nodes are more important in each graph) --> can change over time
4. temporal attention per sequence (important nodes can change over time)
5. produces node embeddings for each graph snapshot based on the spatial attention and temporal attention (past)
6. pooling mechanism to create a single graph embedding from the separate node embeddings
"""

