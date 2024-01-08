from functions.functions import *
from functions.mujoco_functions import *
from lamcts import MCTS
import argparse

import os
import ray

import timeit
from datetime import datetime, timedelta
import numpy as np

parser = argparse.ArgumentParser(description="Process inputs")
parser.add_argument("--func", help="specify the test function")
parser.add_argument("--dims", type=int, help="specify the problem dimensions")
parser.add_argument(
    "--iterations", type=int, help="specify the iterations to collect in the search"
)
parser.add_argument(
    "--cores", type=int, help="specify the number of cores to run MCTS in parallel on"
)
parser.add_argument(
    "--interval", type=int, help="specify the interval to share samples between workers"
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

assert (
    args.iterations % args.interval == 0
), f"Interval {args.interval} does not divide {args.iterations}"

# f = Ackley(dims = 10)
# f = Levy(dims = 10)
# f = Swimmer()
# f = Hopper()
# f = Lunarlanding()

# runtime_env = {
#     "env_vars": {
#         "OMP_NUM_THREADS": "1",
#     }
# }


def get_elapsed_time(start_time):
    delta = datetime.now() - start_time
    milliseconds = delta / timedelta(milliseconds=1)
    return milliseconds


@ray.remote
class remote_MCTS(MCTS):
    pass


def test(iterations, cores, interval):
    f = None
    if args.func == "ackley":
        f = Ackley(dims=args.dims)
    elif args.func == "levy":
        f = Levy(dims=args.dims)
    elif args.func == "lunar":
        f = Lunarlanding()
    elif args.func == "swimmer":
        f = Swimmer()
    elif args.func == "hopper":
        f = Hopper()

    records = []
    start_time = datetime.now()
    ray.init()

    # remote_MCTS = ray.remote(MCTS)

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

    best_local_values = []

    for i in range(0, iterations, interval):
        best_local_values = ray.get(
            [agent.search.remote(iterations=interval) for agent in remote_agents]
        )
        best_local_value = min(best_local_values)
        records.append((get_elapsed_time(start_time), best_local_value))

        samples_ls = ray.get([agent.get_samples.remote() for agent in remote_agents])

        merged_samples = sum(samples_ls, [])
        merged_samples_array = [np.append(x, fx) for (x, fx) in merged_samples]
        merged_samples_array = np.array(merged_samples_array)
        merged_sampels_array = np.unique(merged_samples_array, axis=0)
        merged_samples = merged_sampels_array.tolist()
        merged_samples = [(x[:-1], float(x[-1])) for x in merged_samples]
        print(f"Length of merged samples: {len(merged_samples)}")

        for agent in remote_agents:
            agent.set_samples.remote(merged_samples)
            print(f"Set samples for {agent} with length {len(merged_samples)}")

    samples_ls = ray.get([agent.get_samples.remote() for agent in remote_agents])

    merged_samples = sum(samples_ls, [])
    merged_samples_array = [np.append(x, fx) for (x, fx) in merged_samples]
    merged_samples_array = np.array(merged_samples_array)
    merged_sampels_array = np.unique(merged_samples_array, axis=0)
    merged_samples = merged_sampels_array.tolist()
    merged_samples = [(x[:-1], float(x[-1])) for x in merged_samples]
    print(f"Length of merged samples: {len(merged_samples)}")

    for agent in remote_agents:
        agent.set_samples.remote(merged_samples)
        print(f"Set samples for {agent} with length {len(merged_samples)}")

    f = None
    if args.func == "ackley":
        f = Ackley(dims=args.dims)
    elif args.func == "levy":
        f = Levy(dims=args.dims)
    elif args.func == "lunar":
        f = Lunarlanding()
    elif args.func == "swimmer":
        f = Swimmer()
    elif args.func == "hopper":
        f = Hopper()

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
        init_samples=False,
    )

    agent.set_samples(merged_samples)
    final_best_value = agent.search(1)

    records.append((get_elapsed_time(start_time), final_best_value))

    print(f"Best local values: {best_local_values}")
    print(f"Best final value: {final_best_value}")

    ray.shutdown()

    with open(
        f"results/{args.func}_{args.dims}_{args.iterations}_{args.cores}_{args.interval}.txt",
        "a",
    ) as file:
        file.write(str(records) + "\n")


num_repeat = 3

execution_time_ls = []

for _ in range(num_repeat):
    execution_time = timeit.timeit(
        lambda: test(args.iterations, args.cores, args.interval), number=1
    )
    print(f"Execution time: {execution_time}")
    execution_time_ls.append(execution_time)

print(f"Execution time of {num_repeat} runs: {execution_time_ls}")
execution_time_array = np.array(execution_time_ls)
# total_time = timeit.timeit(lambda: test(args.iterations), number=num_repeat)
print("Total time: ", np.sum(execution_time_array))
print("Avg time: ", np.mean(execution_time_array))
print("Std: ", np.std(execution_time_array))
