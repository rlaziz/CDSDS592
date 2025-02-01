import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the directed graph
def gen_graph():
    G = nx.DiGraph()
    
    edges = {
        "11": ["21", "22"],
        "21": ["31", "32"],
        "22": ["31", "32"],
        "12": ["22", "23"],
        "23": ["31", "32"],
        "31": ["41", "42", "43"],
        "32": ["41", "42", "43"],
        "41": [],
        "42": [],
        "43": []
    }
    
    for node, neighbors in edges.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    return G

# Perform a single random walk
def trial(G, start_nodes, start_p, end_nodes):
    current_node = np.random.choice(start_nodes, p=start_p)

    while current_node not in end_nodes:
        neighbors = list(G.successors(current_node))
        current_node = np.random.choice(neighbors)
    
    return current_node

# Run a single Monte Carlo simulation (tracking error over time)
def run_trials_with_error_tracking(G, start_nodes, start_p, end_nodes, theoretical_pi4, num_trials=10000):
    end_counts = {node: 0 for node in end_nodes}
    errors = []

    for t in range(1, num_trials + 1):
        final_node = trial(G, start_nodes, start_p, end_nodes)
        end_counts[final_node] += 1

        # Compute empirical probabilities
        empirical_pi4 = np.array([end_counts[node] / t for node in end_nodes])
        
        # Compute L2 error
        error = np.linalg.norm(theoretical_pi4 - empirical_pi4, ord=2)
        errors.append(error)

    return errors

# Compute theoretical probabilities using a transition matrix
def compute_theoretical_pi4():
    P = np.array([
        [  0,    0,  0.5,  0.5,    0,    0,    0,    0,    0,    0],  # 11
        [  0,    0,    0,  0.5,  0.5,    0,    0,    0,    0,    0],  # 12
        [  0,    0,    0,    0,    0,  0.5,  0.5,    0,    0,    0],  # 21
        [  0,    0,    0,    0,    0,  0.5,  0.5,    0,    0,    0],  # 22
        [  0,    0,    0,    0,    0,  0.5,  0.5,    0,    0,    0],  # 23
        [  0,    0,    0,    0,    0,    0,    0,  1/3,  1/3,  1/3],  # 31
        [  0,    0,    0,    0,    0,    0,    0,  1/3,  1/3,  1/3],  # 32
        [  0,    0,    0,    0,    0,    0,    0,    1,    0,    0],  # 41
        [  0,    0,    0,    0,    0,    0,    0,    0,    1,    0],  # 42
        [  0,    0,    0,    0,    0,    0,    0,    0,    0,    1],  # 43
    ])

    Q = P[2:7, 2:7]  # Transition matrix for intermediate states
    R = P[2:7, 7:10]  # Transition probabilities to absorbing states
    I = np.eye(Q.shape[0])

    N = np.linalg.inv(I - Q)  # Fundamental matrix
    B = N @ R  # Absorbing probabilities

    results = []
    for pi0 in [[0.5, 0.5], [0.4, 0.6], [0.1, 0.9]]:
        pi0_extended = np.zeros(5)  
        pi0_extended[0] = pi0[0] / 2  
        pi0_extended[1] = pi0[0] / 2  
        pi0_extended[2] = pi0[1] / 2  
        pi0_extended[3] = pi0[1] / 2  
        pi0_extended[4] = 0  

        pi4 = pi0_extended @ B  
        results.append(pi4)

    return results

# Main function
def main():
    G = gen_graph()
    start_nodes = ["11", "12"]
    end_nodes = ["41", "42", "43"]
    
    theoretical_results = compute_theoretical_pi4()

    num_trials = 10000  # Fixed at 10,000 trials

    plt.figure(figsize=(8, 5))

    for i, start_p in enumerate([[0.5, 0.5], [0.4, 0.6], [0.1, 0.9]]):
        theoretical_pi4 = theoretical_results[i]
        errors = run_trials_with_error_tracking(G, start_nodes, start_p, end_nodes, theoretical_pi4, num_trials)

        print(f"π0 = {start_p}")
        print(f"Theoretical π4: {theoretical_pi4}\n")

        plt.plot(range(1, num_trials + 1), errors, label=f"π0 = {start_p}")

    plt.xscale("log")
    plt.xlabel("Number of Trials (log scale)")
    plt.ylabel("Error (L2 Norm)")
    plt.title("Convergence of Monte Carlo π4 to Theoretical π4")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    main()
