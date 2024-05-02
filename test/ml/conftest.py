import pytest
from ray.cluster_utils import Cluster


@pytest.fixture(scope="module")
def ray_cluster():
    import ray

    # create a local cluster with 1 head node and 2 worker nodes
    cluster = Cluster(
        initialize_head=True,
        connect=True,
        head_node_args={"resources": {"num_cpus": 1}},
        shutdown_at_exit=True,
    )
    cluster.add_node(resources={"num_cpus": 2}, num_cpus=2)
    cluster.add_node(resources={"num_cpus": 2}, num_cpus=2)

    yield cluster

    ray.shutdown()
