import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random
from typing import Dict, List, Tuple
import time

class SensorNode:
    def __init__(self, node_id: int, x: float, y: float, node_type: str = "sensor"):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type
        self.cluster_id = None
        self.data_records = []
        self.risk_level = "low"
        self.energy_level = 100.0
        self.transmission_range = 30.0
        
    def generate_sensor_data(self, timestamp: int) -> Dict:
        base_temp = 25.0
        base_humidity = 50.0
        
        if self.node_type == "sensor":
            temperature = base_temp + random.gauss(0, 2)
            humidity = base_humidity + random.gauss(0, 5)
            light = max(0, random.gauss(500, 100))
            voltage = max(3.0, random.gauss(3.3, 0.2))
            
            data = {
                'timestamp': timestamp,
                'node_id': self.node_id,
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'light': round(light, 2),
                'voltage': round(voltage, 2),
                'cluster_id': self.cluster_id,
                'risk_level': self.risk_level
            }
        else:  # gateway
            data = {
                'timestamp': timestamp,
                'node_id': self.node_id,
                'node_type': 'gateway',
                'packets_received': random.randint(50, 100),
                'network_health': random.uniform(0.8, 1.0)
            }
            
        self.data_records.append(data)
        return data
    
    def update_risk_level(self, attack_success_prob: float):
        if attack_success_prob > 0.7:
            self.risk_level = "high"
        elif attack_success_prob > 0.5:
            self.risk_level = "medium"
        else:
            self.risk_level = "low"
    
    def consume_energy(self, operation: str):
        energy_cost = {
            'transmission': 2.0,
            'reception': 1.0,
            'sensing': 0.5,
            'computation': 0.3
        }
        self.energy_level = max(0, self.energy_level - energy_cost.get(operation, 1.0))

class WSNCluster:
    def __init__(self, cluster_id: int, centroid_x: float, centroid_y: float):
        self.cluster_id = cluster_id
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.member_nodes = []
        self.cluster_head = None
        
    def add_node(self, node: SensorNode):
        node.cluster_id = self.cluster_id
        self.member_nodes.append(node)
        
    def set_cluster_head(self, node: SensorNode):
        self.cluster_head = node

class MIAAttacker:
    def __init__(self, position: Tuple[float, float], attack_range: float = 40.0):
        self.x, self.y = position
        self.attack_range = attack_range
        self.detected_nodes = []
        self.attack_success_rate = 0.0
        
    def can_attack_node(self, node: SensorNode) -> bool:
        distance = np.sqrt((node.x - self.x)**2 + (node.y - self.y)**2)
        return distance <= self.attack_range
    
    def perform_mia_attack(self, node: SensorNode, model_confidence: float) -> bool:
        base_success_prob = 0.3
        
        if node.risk_level == "high":
            base_success_prob += 0.4
        elif node.risk_level == "medium":
            base_success_prob += 0.2
            
        if model_confidence > 0.8:
            base_success_prob += 0.3
        elif model_confidence > 0.6:
            base_success_prob += 0.15
            
        distance_factor = 1.0 - (np.sqrt((node.x - self.x)**2 + (node.y - self.y)**2) / self.attack_range)
        base_success_prob += distance_factor * 0.2
        
        success = random.random() < base_success_prob
        
        if success:
            self.detected_nodes.append(node.node_id)
            
        return success

class IoTWSNSimulator:
    def __init__(self, area_width: float = 100.0, area_height: float = 100.0, 
                 num_nodes: int = 20, num_clusters: int = 5):
        self.area_width = area_width
        self.area_height = area_height
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.nodes = []
        self.clusters = []
        self.attackers = []
        self.network_graph = nx.Graph()
        self.simulation_time = 0
        self.data_collection = []
        
        self._initialize_network()
        
    def _initialize_network(self):
        print("Initializing IoT-WSN Network...")
        
        # Create clusters
        cluster_centroids = []
        for i in range(self.num_clusters):
            centroid_x = random.uniform(20, self.area_width - 20)
            centroid_y = random.uniform(20, self.area_height - 20)
            cluster_centroids.append((centroid_x, centroid_y))
            self.clusters.append(WSNCluster(i, centroid_x, centroid_y))
        
        # Create sensor nodes and assign to clusters
        for i in range(self.num_nodes):
            cluster_id = i % self.num_clusters
            centroid_x, centroid_y = cluster_centroids[cluster_id]
            
            # Add some randomness to node positions around cluster centroid
            node_x = centroid_x + random.uniform(-15, 15)
            node_y = centroid_y + random.uniform(-15, 15)
            
            # Ensure nodes stay within area bounds
            node_x = max(0, min(self.area_width, node_x))
            node_y = max(0, min(self.area_height, node_y))
            
            node = SensorNode(i, node_x, node_y)
            self.nodes.append(node)
            self.clusters[cluster_id].add_node(node)
            
            # Set cluster head (first node in each cluster)
            if i % self.num_clusters == 0:
                self.clusters[cluster_id].set_cluster_head(node)
        
        # Create gateway nodes
        gateway_positions = [(10, 10), (90, 10), (10, 90), (90, 90)]
        for i, (gx, gy) in enumerate(gateway_positions):
            gateway = SensorNode(100 + i, gx, gy, "gateway")
            self.nodes.append(gateway)
        
        # Build network graph
        self._build_network_topology()
        
        print(f"Network initialized with {len(self.nodes)} nodes and {len(self.clusters)} clusters")
    
    def _build_network_topology(self):
        self.network_graph.clear()
        
        # Add nodes to graph
        for node in self.nodes:
            self.network_graph.add_node(node.node_id, 
                                      pos=(node.x, node.y),
                                      type=node.node_type,
                                      cluster=node.cluster_id)
        
        # Add edges based on transmission range
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i < j and node1.node_type == "sensor" and node2.node_type == "sensor":
                    distance = np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
                    if distance <= node1.transmission_range:
                        self.network_graph.add_edge(node1.node_id, node2.node_id, 
                                                  weight=distance)
        
        # Connect clusters to gateways
        for cluster in self.clusters:
            if cluster.cluster_head:
                for gateway in self.nodes:
                    if gateway.node_type == "gateway":
                        distance = np.sqrt((cluster.cluster_head.x - gateway.x)**2 + 
                                         (cluster.cluster_head.y - gateway.y)**2)
                        if distance <= cluster.cluster_head.transmission_range * 2:  # Extended range for gateways
                            self.network_graph.add_edge(cluster.cluster_head.node_id, 
                                                      gateway.node_id, 
                                                      weight=distance)
    
    def add_attacker(self, position: Tuple[float, float], attack_range: float = 40.0):
        attacker = MIAAttacker(position, attack_range)
        self.attackers.append(attacker)
        print(f"Added attacker at position {position} with range {attack_range}")
        return attacker
    
    def simulate_time_step(self, apply_defense: bool = False):
        self.simulation_time += 1
        print(f"\n--- Simulation Time Step {self.simulation_time} ---")
        
        # Generate sensor data
        timestamp = int(time.time())
        current_data = []
        
        for node in self.nodes:
            if node.node_type == "sensor":
                data = node.generate_sensor_data(timestamp)
                current_data.append(data)
                node.consume_energy('sensing')
        
        self.data_collection.extend(current_data)
        
        # Simulate MIA attacks
        attack_results = self._simulate_mia_attacks(apply_defense)
        
        # Update network topology (occasionally)
        if self.simulation_time % 10 == 0:
            self._build_network_topology()
        
        return current_data, attack_results
    
    def _simulate_mia_attacks(self, apply_defense: bool) -> Dict:
        attack_results = {
            'total_attempts': 0,
            'successful_attacks': 0,
            'compromised_nodes': [],
            'attack_success_rate': 0.0
        }
        
        for attacker in self.attackers:
            for node in self.nodes:
                if node.node_type == "sensor" and attacker.can_attack_node(node):
                    attack_results['total_attempts'] += 1
                    
                    # Simulate model confidence (higher for edge nodes)
                    is_edge_node = self._is_edge_node(node)
                    base_confidence = 0.7 if is_edge_node else 0.5
                    
                    if apply_defense:
                        # Apply AOM-SFT defense: reduce confidence
                        base_confidence = max(0.3, base_confidence - 0.4)
                    
                    success = attacker.perform_mia_attack(node, base_confidence)
                    
                    if success:
                        attack_results['successful_attacks'] += 1
                        attack_results['compromised_nodes'].append(node.node_id)
                        node.update_risk_level(0.8)
                    else:
                        node.update_risk_level(0.3)
        
        if attack_results['total_attempts'] > 0:
            attack_results['attack_success_rate'] = (
                attack_results['successful_attacks'] / attack_results['total_attempts']
            )
        
        attacker.attack_success_rate = attack_results['attack_success_rate']
        
        print(f"MIA Attacks: {attack_results['successful_attacks']}/{attack_results['total_attempts']} "
              f"successful ({attack_results['attack_success_rate']:.2%})")
        
        return attack_results
    
    def _is_edge_node(self, node: SensorNode) -> bool:
        node_degree = self.network_graph.degree(node.node_id)
        return node_degree <= 2
    
    def get_network_stats(self) -> Dict:
        total_nodes = len([n for n in self.nodes if n.node_type == "sensor"])
        high_risk_nodes = len([n for n in self.nodes if n.risk_level == "high"])
        medium_risk_nodes = len([n for n in self.nodes if n.risk_level == "medium"])
        low_risk_nodes = len([n for n in self.nodes if n.risk_level == "low"])
        
        avg_energy = np.mean([n.energy_level for n in self.nodes if n.node_type == "sensor"])
        
        network_connectivity = nx.density(self.network_graph)
        
        return {
            'total_sensor_nodes': total_nodes,
            'high_risk_nodes': high_risk_nodes,
            'medium_risk_nodes': medium_risk_nodes,
            'low_risk_nodes': low_risk_nodes,
            'high_risk_percentage': (high_risk_nodes / total_nodes) * 100,
            'average_energy': avg_energy,
            'network_connectivity': network_connectivity,
            'total_data_points': len(self.data_collection)
        }
    
    def visualize_network(self, show_attack: bool = True, filename: str = None):
        plt.figure(figsize=(12, 10))
        
        # Create position dictionary for plotting
        pos = {node.node_id: (node.x, node.y) for node in self.nodes}
        
        # Define colors based on node properties
        node_colors = []
        node_sizes = []
        
        for node in self.nodes:
            if node.node_type == "gateway":
                node_colors.append('red')
                node_sizes.append(200)
            else:
                if node.risk_level == "high":
                    node_colors.append('darkred')
                elif node.risk_level == "medium":
                    node_colors.append('orange')
                else:
                    node_colors.append('lightblue')
                node_sizes.append(100)
        
        # Draw the network
        nx.draw(self.network_graph, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                edge_color='gray',
                width=1.0,
                alpha=0.7,
                with_labels=True,
                font_size=8,
                font_weight='bold')
        
        # Draw cluster areas
        for cluster in self.clusters:
            circle = plt.Circle((cluster.centroid_x, cluster.centroid_y), 
                              20, fill=False, linestyle='--', 
                              color='blue', alpha=0.3)
            plt.gca().add_patch(circle)
            plt.text(cluster.centroid_x, cluster.centroid_y, 
                    f'C{cluster.cluster_id}', fontsize=10, 
                    ha='center', va='center', color='blue')
        
        # Draw attackers
        if show_attack and self.attackers:
            for attacker in self.attackers:
                plt.scatter(attacker.x, attacker.y, 
                          color='black', marker='*', s=300, 
                          label=f'Attacker (Success: {attacker.attack_success_rate:.2%})')
                
                # Draw attack range
                attack_circle = plt.Circle((attacker.x, attacker.y), 
                                         attacker.attack_range, 
                                         fill=False, color='red', 
                                         linestyle=':', alpha=0.5)
                plt.gca().add_patch(attack_circle)
        
        plt.title(f'IoT-WSN Simulation\nTime Step: {self.simulation_time}')
        plt.xlim(0, self.area_width)
        plt.ylim(0, self.area_height)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Low Risk'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Medium Risk'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10, label='High Risk'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Gateway'),
        ]
        
        if show_attack and self.attackers:
            legend_elements.append(
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=15, label='Attacker')
            )
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add statistics text box
        stats = self.get_network_stats()
        stats_text = (f"Network Statistics:\n"
                     f"Sensor Nodes: {stats['total_sensor_nodes']}\n"
                     f"High Risk: {stats['high_risk_nodes']} ({stats['high_risk_percentage']:.1f}%)\n"
                     f"Avg Energy: {stats['average_energy']:.1f}%\n"
                     f"Data Points: {stats['total_data_points']}")
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved as {filename}")
        
        plt.tight_layout()
        plt.show()
    
    def run_simulation(self, num_steps: int = 50, apply_defense: bool = False):
        print(f"Starting simulation for {num_steps} time steps...")
        print(f"Defense mechanism: {'AOM-SFT Enabled' if apply_defense else 'No Defense'}")
        
        all_attack_results = []
        network_stats_history = []
        
        for step in range(num_steps):
            current_data, attack_results = self.simulate_time_step(apply_defense)
            stats = self.get_network_stats()
            
            all_attack_results.append(attack_results)
            network_stats_history.append(stats)
            
            if (step + 1) % 10 == 0:
                print(f"Completed {step + 1}/{num_steps} time steps")
        
        print(f"\nSimulation completed after {num_steps} time steps")
        
        # Generate summary report
        self._generate_simulation_report(all_attack_results, network_stats_history, apply_defense)
        
        return all_attack_results, network_stats_history
    
    def _generate_simulation_report(self, attack_results, network_stats, apply_defense):
        print("\n" + "="*50)
        print("SIMULATION SUMMARY REPORT")
        print("="*50)
        
        total_attempts = sum([r['total_attempts'] for r in attack_results])
        total_successes = sum([r['successful_attacks'] for r in attack_results])
        
        if total_attempts > 0:
            overall_success_rate = total_successes / total_attempts
        else:
            overall_success_rate = 0.0
        
        final_stats = network_stats[-1]
        
        print(f"Defense Mechanism: {'AOM-SFT' if apply_defense else 'None'}")
        print(f"Total MIA Attempts: {total_attempts}")
        print(f"Successful Attacks: {total_successes}")
        print(f"Overall Attack Success Rate: {overall_success_rate:.2%}")
        print(f"High Risk Nodes: {final_stats['high_risk_nodes']} ({final_stats['high_risk_percentage']:.1f}%)")
        print(f"Network Connectivity: {final_stats['network_connectivity']:.3f}")
        print(f"Total Data Collected: {final_stats['total_data_points']} points")
        
        if apply_defense:
            risk_reduction = 35 - final_stats['high_risk_percentage']  # Assuming 35% baseline
            print(f"Risk Reduction: {risk_reduction:.1f}%")
        
        print("="*50)

def demo_simulation():
    print("IoT-WSN Simulation Demo")
    print("This demo creates a 100x100 unit area with 20 sensor nodes in 5 clusters")
    
    # Create simulator
    simulator = IoTWSNSimulator(num_nodes=20, num_clusters=5)
    
    # Add attacker
    simulator.add_attacker((90, 10), attack_range=40)
    
    # Run simulation without defense
    print("\n1. Running simulation WITHOUT defense...")
    attack_results_no_defense, stats_no_defense = simulator.run_simulation(
        num_steps=20, apply_defense=False
    )
    
    # Visualize final state without defense
    simulator.visualize_network(show_attack=True, filename="network_no_defense.png")
    
    # Reset for defense simulation
    simulator_defense = IoTWSNSimulator(num_nodes=20, num_clusters=5)
    simulator_defense.add_attacker((90, 10), attack_range=40)
    
    # Run simulation with defense
    print("\n2. Running simulation WITH AOM-SFT defense...")
    attack_results_defense, stats_defense = simulator_defense.run_simulation(
        num_steps=20, apply_defense=True
    )
    
    # Visualize final state with defense
    simulator_defense.visualize_network(show_attack=True, filename="network_with_defense.png")
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    no_defense_success = sum([r['successful_attacks'] for r in attack_results_no_defense])
    defense_success = sum([r['successful_attacks'] for r in attack_results_defense])
    
    no_defense_attempts = sum([r['total_attempts'] for r in attack_results_no_defense])
    defense_attempts = sum([r['total_attempts'] for r in attack_results_defense])
    
    print(f"Without Defense: {no_defense_success}/{no_defense_attempts} successful attacks")
    print(f"With AOM-SFT: {defense_success}/{defense_attempts} successful attacks")
    
    if no_defense_attempts > 0 and defense_attempts > 0:
        no_defense_rate = no_defense_success / no_defense_attempts
        defense_rate = defense_success / defense_attempts
        improvement = (no_defense_rate - defense_rate) / no_defense_rate * 100
        
        print(f"Attack Success Rate Reduction: {improvement:.1f}%")

if __name__ == "__main__":
    demo_simulation()
