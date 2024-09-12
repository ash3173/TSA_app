import pandas as pd
import numpy as np
def k_hop_neighbors(edge_data, node, k):
    neighbors = set()
    current_level = {node}
    for _ in range(k):
        next_level = set()
        for n in current_level:
            next_level.update(edge_data[(edge_data['source'] == n) & (edge_data['value'] == 1)]['target'])
        current_level = next_level - neighbors
        neighbors.update(current_level)
    return neighbors

def aggregate_node_with_neighbors(node_data, edge_data, k, target_node, target_feature):
    aggregated_series = []
    
    for timestamp in node_data['timestamp'].unique():
        # Get the current feature value of the target node at the given timestamp
        node_value = node_data[(node_data['timestamp'] == timestamp) & 
                               (node_data['node'] == target_node) & 
                               (node_data['feature'] == target_feature)]['value'].values[0]
        
        # Get k-hop neighbors of the target node
        neighbors = k_hop_neighbors(edge_data, target_node, k)
        
        # Compute the mean value of neighbors for the target feature
        neighbor_values = node_data[(node_data['timestamp'] == timestamp) & 
                                    (node_data['node'].isin(neighbors)) & 
                                    (node_data['feature'] == target_feature)]['value']
        agg_value = neighbor_values.mean() if not neighbor_values.empty else 0
        
        # Append the current node's value and the aggregated neighbor value to the series
        aggregated_series.append([node_value, agg_value])
    
    return aggregated_series

def create_sequences_for_prediction(aggregated_series, sequence_length):
    X, y = [], []
    
    for i in range(len(aggregated_series) - sequence_length):
        seq_x = aggregated_series[i:i + sequence_length]
        X.append(seq_x)
        
        # The target value is the node's feature value at the next time step
        y.append(aggregated_series[i + sequence_length][0])
    
    return np.array(X), np.array(y)




