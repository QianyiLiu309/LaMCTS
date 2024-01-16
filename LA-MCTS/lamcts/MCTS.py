# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
from .utils import latin_hypercube, from_unit_cube
from torch.quasirandom import SobolEngine
import torch
import ray

class MCTS:
    #############################################

    def __init__(
        self,
        lb,
        ub,
        dims,
        ninits,
        func,
        Cp=1,
        leaf_size=20,
        kernel_type="rbf",
        gamma_type="auto",
        use_botorch=False,
        init_samples=True,
        forget=False,
        leaf_parallel=False,
    ):
        self.dims = dims
        self.samples = []
        self.nodes = []
        self.Cp = Cp
        self.lb = lb
        self.ub = ub
        self.ninits = ninits
        self.func = func
        self.curt_best_value = float("-inf")
        self.curt_best_sample = None
        self.best_value_trace = []
        self.sample_counter = 0
        self.visualization = False
        self.use_botorch = use_botorch
        self.forget = forget
        self.leaf_parallel = leaf_parallel

        self.LEAF_SAMPLE_SIZE = leaf_size
        self.kernel_type = kernel_type
        self.gamma_type = gamma_type

        self.solver_type = "bo"  # solver can be 'bo' or 'turbo'

        print("gamma_type:", gamma_type)

        # we start the most basic form of the tree, 3 nodes and height = 1
        root = Node(
            parent=None,
            dims=self.dims,
            reset_id=True,
            kernel_type=self.kernel_type,
            gamma_type=self.gamma_type,
            use_botorch=self.use_botorch,
        )
        self.nodes.append(root)

        self.ROOT = root
        self.CURT = self.ROOT

        if init_samples:
            self.init_train()

        self.iteration_counter = 0
        self.improvement_made = False

        self.num_samples = []
        self.leaf_scores = []

    def populate_training_data(self):
        # only keep root
        if self.forget and self.improvement_made:
            samples_to_keep = self.select_samples_to_keep()
        else:
            samples_to_keep = None

        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root = Node(
            parent=None,
            dims=self.dims,
            reset_id=True,
            kernel_type=self.kernel_type,
            gamma_type=self.gamma_type,
            use_botorch=self.use_botorch,
        )
        self.nodes.append(new_root)

        self.ROOT = new_root
        self.CURT = self.ROOT
        # self.ROOT.update_bag(self.samples)
        if samples_to_keep is None:
            self.ROOT.update_bag(self.samples)
        else:
            self.ROOT.update_bag(samples_to_keep)
            self.samples = samples_to_keep

    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if (
                node.is_leaf() == True
                and len(node.bag) > self.LEAF_SAMPLE_SIZE
                and node.is_svm_splittable == True
            ):
                status.append(True)
            else:
                status.append(False)
        return np.array(status)

    def get_split_idx(self):
        split_by_samples = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return split_by_samples

    def is_splitable(self):
        # check if any of the nodes is splittable
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False

    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes) == 1

        while self.is_splitable():
            # splittable node indices
            to_split = self.get_split_idx()
            # print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[
                    nidx
                ]  # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                # creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data) > 0
                good_kid = Node(
                    parent=parent,
                    dims=self.dims,
                    reset_id=False,
                    kernel_type=self.kernel_type,
                    gamma_type=self.gamma_type,
                    use_botorch=self.use_botorch,
                )
                bad_kid = Node(
                    parent=parent,
                    dims=self.dims,
                    reset_id=False,
                    kernel_type=self.kernel_type,
                    gamma_type=self.gamma_type,
                    use_botorch=self.use_botorch,
                )
                good_kid.update_bag(good_kid_data)
                bad_kid.update_bag(bad_kid_data)

                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)

                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

            # print("continue split:", self.is_splitable())

        self.print_tree()

    def collect_samples(self, sample, value=None):
        # print(sample.shape)
        # exit()
        # TODO: to perform some checks here
        if value == None:
            value = self.func(sample) * -1

        print(value)
        if value > self.curt_best_value:
            self.curt_best_value = value
            self.curt_best_sample = sample
            self.best_value_trace.append((value, self.sample_counter))
            self.improvement_made = True
        self.sample_counter += 1
        self.samples.append((sample, value))
        return value

    def init_train(self):
        print(f"Init training with {self.ninits} points")
        # here we use latin hyper space to generate init samples in the search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)

        for point in init_points:
            print("init point:", point)
            self.collect_samples(point)

        print(
            "=" * 10
            + "collect "
            + str(len(self.samples))
            + " points for initializing MCTS"
            + "=" * 10
        )
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("=" * 58)

    def print_tree(self):
        print("-" * 100)
        for node in self.nodes:
            print(node)
        print("-" * 100)

    def reset_to_root(self):
        self.CURT = self.ROOT

    def load_agent(self):
        node_path = "mcts_agent"
        if os.path.isfile(node_path) == True:
            with open(node_path, "rb") as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples), " samples")

    def dump_agent(self):
        node_path = "mcts_agent"
        print("dumping the agent.....")
        with open(node_path, "wb") as outfile:
            pickle.dump(self, outfile)

    def dump_samples(self):
        sample_path = "samples_" + str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)

    def dump_trace(self):
        trace_path = "best_values_trace"
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + "\n")

    # def greedy_select(self):
    #     self.reset_to_root()
    #     curt_node = self.ROOT
    #     path = []
    #     if self.visualization == True:
    #         curt_node.plot_samples_and_boundary(self.func)
    #     while curt_node.is_leaf() == False:
    #         UCT = []
    #         for i in curt_node.kids:
    #             UCT.append(i.get_xbar())
    #         choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[
    #             0
    #         ]
    #         path.append((curt_node, choice))
    #         curt_node = curt_node.kids[choice]
    #         if curt_node.is_leaf() == False and self.visualization == True:
    #             curt_node.plot_samples_and_boundary(self.func)
    #         print("=>", curt_node.get_name(), end=" ")
    #     print("")
    #     return curt_node, path

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path = []

        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            choices = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)
            choice = choices[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            print("=>", curt_node.get_name(), end=" ")
        print("")
        return curt_node, path
    
    def select_samples_to_keep(self):
        # if not self.ROOT.is_leaf() and not self.ROOT.kids[0].is_leaf() and not self.ROOT.kids[1].is_leaf():
        #     print(f"node 1 has {len(self.ROOT.kids[0].bag)} samples")
        #     print(f"node 2 has {len(self.ROOT.kids[1].bag)} samples")
        #     print(f"node 3 has {len(self.ROOT.kids[0].kids[0].bag)} samples")
        #     print(f"node 4 has {len(self.ROOT.kids[0].kids[1].bag)} samples")
        #     print(f"node 5 has {len(self.ROOT.kids[1].kids[0].bag)} samples")
        #     print(f"node 6 has {len(self.ROOT.kids[1].kids[1].bag)} samples")
        #     samples_to_keep = self.ROOT.kids[0].bag + self.ROOT.kids[1].kids[0].bag
        #     print(f"keep {len(samples_to_keep)} samples, with total {len(self.samples)} samples")
        # else:
        #     print(f"Keep all samples")
        #     samples_to_keep = None
        # return samples_to_keep

        if self.ROOT.is_leaf():
            print(f"Keep all samples")
            return None

        samples_to_keep = []        
        curt_node = self.ROOT

        while not curt_node.kids[1].is_leaf():
            samples_to_keep += curt_node.kids[0].bag
            curt_node = curt_node.kids[1]
        print(f"Remove samples from node {curt_node.kids[1].get_name()}")
        samples_to_keep += curt_node.kids[0].bag
        print(curt_node.classifier.svm.support_vectors_.shape)
        for sample in curt_node.kids[1].bag:
            # print(sample[0])
            if sample[0] in curt_node.classifier.svm.support_vectors_:
                samples_to_keep.append(sample)
        print(f"keep {len(samples_to_keep)} samples, with total {len(self.samples)} samples")
        return samples_to_keep


    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar * curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n += 1
            curt_node = curt_node.parent

    def search(self, iterations):
        if self.leaf_parallel:
            ray.init()
        for idx in range(self.sample_counter, self.sample_counter + iterations):
            print("")
            print("=" * 10)
            print("iteration:", self.iteration_counter)
            self.iteration_counter += 1
            print("=" * 10)
            self.dynamic_treeify()
            self.improvement_made = False
            leaf, path = self.select()
            for i in range(0, 1):
                if self.solver_type == "bo" and self.use_botorch == True:
                    samples = leaf.propose_samples_botorch(
                        1, path, self.lb, self.ub, self.samples
                    )
                    print(f"samples: {samples.shape}")
                elif self.solver_type == "bo":
                    samples = leaf.propose_samples_bo(
                        1, path, self.lb, self.ub, self.samples
                    )
                elif self.solver_type == "turbo":
                    samples, values = leaf.propose_samples_turbo(10000, path, self.func)
                else:
                    raise Exception("solver not implemented")
                
                if self.leaf_parallel:
                    @ray.remote(num_cpus=1)
                    def evaluation(sample):
                        return sample, self.func(sample) * -1
                    
                    remote_evaluations = [evaluation.remote(sample) for sample in samples]
                    print("remote evaluations:", len(remote_evaluations))
                    evaluation_results = ray.get(remote_evaluations) 
                    for sample, value in evaluation_results:
                        if value > self.curt_best_value:
                            self.curt_best_value = value
                            self.curt_best_sample = sample
                            self.best_value_trace.append((value, self.sample_counter))
                            self.improvement_made = True
                        self.sample_counter += 1
                        self.samples.append((sample, value))
                        self.func.tracker.track(value * -1)

                        self.backpropogate(leaf, value)

                else:
                    for idx in range(0, len(samples)):
                        # here, this is sequential
                        if self.solver_type == "bo" and self.use_botorch == True:
                            value = self.collect_samples(samples[idx])
                        elif self.solver_type == "bo":
                            value = self.collect_samples(samples[idx])
                        elif self.solver_type == "turbo":
                            value = self.collect_samples(samples[idx], values[idx])
                        else:
                            raise Exception("solver not implemented")

                        self.backpropogate(leaf, value)
            print("total samples:", len(self.samples))
            print("current best f(x):", np.absolute(self.curt_best_value))
            # print("current best x:", np.around(self.curt_best_sample, decimals=1) )
            print("current best x:", self.curt_best_sample)
            self.num_samples.append(len(self.samples))
            self.leaf_scores.append(leaf.x_bar)
        name = str(iterations)
        if self.use_botorch:
            name += "_botorch"
        if self.forget:
            name += "_forget"
        self.func.tracker.dump_trace(name=name)
        # self.dump_trace()
        if self.leaf_parallel:
            ray.shutdown()

        with open(
            f"results/{name}.txt",
            "a",
        ) as file:
            file.write(str(self.num_samples) + "\n")
            file.write(str(self.leaf_scores) + "\n")

        return np.absolute(self.curt_best_value)

    def get_samples(self):
        print("get samples:", len(self.samples))
        return self.samples

    def set_samples(self, samples, start_index=0):
        # self.samples = samples
        # self.sample_counter = len(samples)
        # print("set samples:", len(samples))
        # print("set sample counter:", self.sample_counter)
        self.samples.clear()
        self.samples_counter = 0
        self.curt_best_value = float("-inf")
        self.curt_best_sample = None
        for sample, value in samples:
            if value > self.curt_best_value:
                self.curt_best_value = value
                self.curt_best_sample = sample
            self.sample_counter += 1
            self.samples.append((sample, value))
        print("set samples:", len(self.samples))
