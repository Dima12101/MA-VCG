import random
import uuid
import math
import networkx as nx

from typing import List

from data import IoTTask, ResourceRequest

# --------------------------------------------------------------------------- #
#                                 AGENTS                                      #
# --------------------------------------------------------------------------- #
class Agent:
    """Base class."""
    def __init__(self, agent_id: int):
        self.id = agent_id


# ----------------------------- IoT DEVICES ----------------------------------#
class IoTDevice(Agent):
    """Resource consumer."""
    def __init__(self, agent_id: int, dev_type: str):
        super().__init__(agent_id)
        self.type = dev_type                     # sensor / camera / etc.
        self.energy_budget = 100.0               # percentage
        self.pending_tasks: List[IoTTask] = []

    # -------- realistic task generation -------- #
    def generate_task(self) -> IoTTask:
        if self.type == "sensor":
            cpu = random.uniform(0.1, 0.5)
            memory = random.uniform(0.1, 1.0)
            deadline = random.uniform(1.0, 5.0)
            data_size = random.uniform(1.0, 10.0)
        elif self.type == "camera":
            cpu = random.uniform(2.0, 8.0)
            memory = random.uniform(4.0, 16.0)
            deadline = random.uniform(0.5, 2.0)
            data_size = random.uniform(10.0, 100.0)
        else:                                   # generic
            cpu = random.uniform(0.5, 2.0)
            memory = random.uniform(1.0, 4.0)
            deadline = random.uniform(1.0, 10.0)
            data_size = random.uniform(1.0, 20.0)

        task = IoTTask(
            task_id=str(uuid.uuid4())[:8],
            cpu=cpu,
            memory=memory,
            deadline=deadline,
            data_size=data_size,
            priority=random.randint(1, 5),
        )
        self.pending_tasks.append(task)
        return task

    # -------- utility (bid) estimation -------- #
    def utility_for_task(self, task: IoTTask, expected_completion: float,
                         energy_cost: float) -> float:
        """Example exponential utility decreasing with delay + energy."""
        time_penalty = math.exp(-expected_completion / task.deadline)
        base = task.priority * (task.cpu + task.memory)                # importance
        return base * time_penalty - energy_cost

    def build_requests(self, network: nx.Graph,
                       providers: List["EdgeNode"]) -> List[ResourceRequest]:
        """Create bids for each pending task against the current provider set."""
        requests: List[ResourceRequest] = []
        for task in list(self.pending_tasks):
            # rough upper-bound on completion time (compute + net delay)
            best_latency = min(
                network[self.id][p.id]['weight'] for p in providers
                if network.has_edge(self.id, p.id)
            )
            expected_completion = best_latency + task.cpu / 10          # simplification
            energy_cost = best_latency * 0.1
            utility = self.utility_for_task(task, expected_completion, energy_cost)
            requests.append(ResourceRequest(self.id, task, utility))
        return requests


# ------------------------------ EDGE NODES ----------------------------------#
class EdgeNode(Agent):
    """Resource provider."""
    def __init__(self, agent_id: int):
        super().__init__(agent_id)
        # capacities
        self.capacity = {
            "cpu": random.uniform(20.0, 40.0),
            "memory": random.uniform(32.0, 64.0),
        }
        self.available = self.capacity.copy()
        # pricing parameters
        self.base_price = {"cpu": 0.2, "memory": 0.05}      # $/unit
        self.energy_price = 0.01                            # $/J
        self.power_per_cpu = 2.0                            # J per CPU-unit

    # -------- cost (bid) estimation -------- #
    def cost_for_request(self, req: ResourceRequest,
                         network: nx.Graph) -> float | None:
        cpu, mem = req.task.cpu, req.task.memory
        if cpu > self.available["cpu"] or mem > self.available["memory"]:
            return None

        # base cost
        cost = (cpu * self.base_price["cpu"] +
                mem * self.base_price["memory"])

        # load multiplier
        load = 1 - (self.available["cpu"] / self.capacity["cpu"])
        cost *= 1 + load ** 2

        # energy
        cost += cpu * self.power_per_cpu * self.energy_price

        # communication delay cost (latency as monetary penalty)
        if network.has_edge(self.id, req.device_id):
            delay = network[self.id][req.device_id]['weight']
        else:
            try:
                delay = nx.shortest_path_length(network, self.id,
                                                 req.device_id, weight='weight')
            except nx.NetworkXNoPath:
                return None
        cost += delay * 0.1
        return cost

    # book-keeping
    def allocate(self, req: ResourceRequest):
        self.available["cpu"] -= req.task.cpu
        self.available["memory"] -= req.task.memory