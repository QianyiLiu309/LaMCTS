from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS
import argparse

import os
import ray

import timeit

parser = argparse.ArgumentParser(description="Process inputs")
parser.add_argument("--func", help="specify the test function")
parser.add_argument("--dims", type=int, help="specify the problem dimensions")
parser.add_argument(
    "--iterations", type=int, help="specify the iterations to collect in the search"
)
parser.add_argument(
    "--cores", type=int, help="specify the number of cores to run MCTS in parallel on"
)


args = parser.parse_args()

f = None
iteration = 0
if args.func == "ackley":
    assert args.dims > 0
    f = Ackley(dims=args.dims)
elif args.func == "levy":
    assert args.dims > 0
    f = Levy(dims=args.dims)
elif args.func == "lunar":
    f = Lunarlanding()
elif args.func == "swimmer":
    f = Swimmer()
elif args.func == "hopper":
    f = Hopper()
else:
    print("function not defined")
    os._exit(1)

assert f is not None
assert args.iterations > 0

assert (
    args.cores <= os.cpu_count()
), f"Too many cores requested: {args.cores} > {os.cpu_count()}"


# f = Ackley(dims = 10)
# f = Levy(dims = 10)
# f = Swimmer()
# f = Hopper()
# f = Lunarlanding()


def test(iterations, cores):
    ray.init()

    remote_MCTS = ray.remote(MCTS)

    remote_agents = [
        remote_MCTS.remote(
            lb=f.lb,  # the lower bound of each problem dimensions
            ub=f.ub,  # the upper bound of each problem dimensions
            dims=f.dims,  # the problem dimensions
            ninits=f.ninits,  # the number of random samples used in initializations
            func=f,  # function object to be optimized
            Cp=f.Cp,  # Cp for MCTS
            leaf_size=f.leaf_size,  # tree leaf size
            kernel_type=f.kernel_type,  # SVM configruation
            gamma_type=f.gamma_type,  # SVM configruation
        )
        for _ in range(cores)
    ]
    print(remote_agents)

    best_local_values = ray.get(
        [agent.search.remote(iterations=iterations) for agent in remote_agents]
    )

    samples_ls = ray.get([agent.get_samples.remote() for agent in remote_agents])

    merged_samples = sum(samples_ls, [])
    print(f"Length of merged samples: {len(merged_samples)}")

    agent = MCTS(
        lb=f.lb,  # the lower bound of each problem dimensions
        ub=f.ub,  # the upper bound of each problem dimensions
        dims=f.dims,  # the problem dimensions
        ninits=0,  # the number of random samples used in initializations
        func=f,  # function object to be optimized
        Cp=f.Cp,  # Cp for MCTS
        leaf_size=f.leaf_size,  # tree leaf size
        kernel_type=f.kernel_type,  # SVM configruation
        gamma_type=f.gamma_type,  # SVM configruation
    )

    agent.set_samples(merged_samples)
    final_best_value = agent.search(len(merged_samples) + 1)

    print(f"Best local values: {best_local_values}")
    print(f"Best final value: {final_best_value}")

    ray.shutdown()


num_repeat = 5
total_time = timeit.timeit(lambda: test(args.iterations, args.cores), number=num_repeat)
print("Total time: ", total_time)
print("Avg time: ", total_time / num_repeat)
