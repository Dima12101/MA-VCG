import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from typing import Dict, List

from agents import EdgeNode, IoTDevice
from auction import Auctioneer

# --------------------------------------------------------------------------- #
#                        EDGE-COMPUTING SYSTEM SIMULATOR                      #
# --------------------------------------------------------------------------- #
class EdgeComputingSystem:
    def __init__(self, n_nodes: int = 10, n_devices: int = 20):
        self.network = nx.Graph()
        self.nodes: List[EdgeNode] = [EdgeNode(i) for i in range(n_nodes)]
        self.devices: List[IoTDevice] = [
            IoTDevice(n_nodes + i, random.choice(["sensor", "camera", "generic"]))
            for i in range(n_devices)
        ]
        self.auctioneer = Auctioneer()
        self._init_topology()
        self._visualize()

    # ---------------------- topology construction --------------------------- #
    def _init_topology(self):
        # add vertices
        for n in self.nodes + self.devices:
            if isinstance(n, EdgeNode):
                self.network.add_node(n.id, type="node")
            else:
                self.network.add_node(n.id, type="device")

        # node-to-node links
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if random.random() < .4: # probability 40%
                    w = random.uniform(1.0, 5.0)      # latency (ms)
                    self.network.add_edge(self.nodes[i].id,
                                          self.nodes[j].id,
                                          weight=w)
        # device-to-node links
        for d in self.devices:
            connected = random.sample(self.nodes,
                                       k=random.randint(1, min(3, len(self.nodes))))
            for n in connected:
                w = random.uniform(0.5, 2.0)
                self.network.add_edge(d.id, n.id, weight=w)

    def _color_for_device(self, device: IoTDevice):
        match device.type:
            case "sensor":
                return "bisque"
            case "camera":
                return "pink"
            case "generic":
                return "silver"

    def _visualize(self):
        G = self.network

        # Colors
        node_colors = ['lightskyblue' for n in self.nodes] + [self._color_for_device(d) for d in self.devices]

        # node_colors = ['skyblue' if G.nodes[n]['type'] == 'node' else 'orange' for n in G.nodes]
        node_sizes = [800 if G.nodes[n]['type'] == 'node' else 400 for n in G.nodes]

        # Position
        pos = nx.spring_layout(G, k=1.2, seed=42)

        plt.figure(figsize=(12, 10))
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            width=0.5,
            alpha=0.85,
            font_size=10,
            font_weight='bold',
            edge_color='gray'
        )

        # Legends
        legend_elements = [
            Patch(facecolor='lightskyblue', edgecolor='k', label='Edge Node'),
            Patch(facecolor='bisque', edgecolor='k', label='IoT Device (sensor)'),
            Patch(facecolor='pink', edgecolor='k', label='IoT Device (camera)'),
            Patch(facecolor='silver', edgecolor='k', label='IoT Device (generic)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('network.png')


    # --------------------------- metrics ------------------------------------ #
    @staticmethod
    def jain_index(values: List[float]) -> float:
        n = len(values)
        if n == 0:
            return 1.0
        s = sum(values)
        sq = sum(v ** 2 for v in values)
        if sq == 0:
            return 1.0
        return s ** 2 / (n * sq)

    # -------------------------- main loop ----------------------------------- #
    def run(self, n_rounds: int = 30):
        metrics = {
            "social_welfare": [],
            "allocation_eff": [],
            "fairness": [],
        }

        for _ in range(n_rounds):
            # 1. devices create tasks & requests
            all_requests = []
            for dev in self.devices:
                dev.generate_task()                      # one per round
                all_requests.extend(dev.build_requests(self.network, self.nodes))

            # 2. auctioneer allocates
            allocations, _ = self.auctioneer.run(all_requests, self.nodes, self.network)

            # 3. apply allocations + compute welfare
            welfare = 0.0
            per_device_allocated: Dict[int, int] = {d.id: 0 for d in self.devices}

            for req, winner, price in allocations:
                # winner side
                winner.allocate(req)
                # consumer side
                per_device_allocated[req.device_id] += 1
                welfare += req.bid_value - price
                # remove task from device queue
                dev = next(d for d in self.devices if d.id == req.device_id)
                dev.pending_tasks.remove(req.task)

            # 4. metrics
            total_tasks = sum(len(d.pending_tasks) + per_device_allocated[d.id]
                              for d in self.devices)
            alloc_eff = (sum(per_device_allocated.values()) /
                         total_tasks) if total_tasks else 1.0
            fairness = self.jain_index(list(per_device_allocated.values()))
            metrics["social_welfare"].append(welfare)
            metrics["allocation_eff"].append(alloc_eff)
            metrics["fairness"].append(fairness)

        return {
            "avg_welfare": float(np.mean(metrics["social_welfare"])),
            "avg_efficiency": float(np.mean(metrics["allocation_eff"])),
            "avg_fairness": float(np.mean(metrics["fairness"])),
        }
