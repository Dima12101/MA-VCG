import networkx as nx

from typing import List, Tuple

from data import ResourceRequest
from agents import EdgeNode

# --------------------------------------------------------------------------- #
#                              AUCTIONEER (VICKREY)                           #
# --------------------------------------------------------------------------- #
class Auctioneer:
    """
    Single-item (per request) Vickrey auction.
    Providers (=edge nodes) submit cost bids.
    Winner delivers the resource and receives the 2-nd lowest bid as payment.
    """

    def run(self, requests: List[ResourceRequest],
            providers: List[EdgeNode],
            network: nx.Graph) -> Tuple[List[Tuple[ResourceRequest, EdgeNode, float]],
                                        List[Tuple[ResourceRequest, str]]]:
        """Return (allocations, rejections)."""
        allocations = []
        rejections = []

        for req in requests:
            bids = []
            for p in providers:
                cost = p.cost_for_request(req, network)
                if cost is not None:
                    bids.append((p, cost))
            if len(bids) < 1:
                rejections.append((req, "no provider"))
                continue

            bids.sort(key=lambda x: x[1])              # ascending (cost)
            winner, win_cost = bids[0]
            pay_price = bids[1][1] if len(bids) > 1 else win_cost  # 2-nd price
            allocations.append((req, winner, pay_price))

        return allocations, rejections