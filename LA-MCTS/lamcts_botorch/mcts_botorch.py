from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from botorch.test_functions import Ackley

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.constraints import Interval

import torch
from torch import Tensor


from Node import Node


class LaMCTS_GP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(
        self,
        all_train_X: Tensor,  # all the X samples collected so far, tensor shape: (num_samples, dims)
        all_train_Y: Tensor,  # tensor shape: (num_samples, 1)
        train_Yvar: Optional[Tensor] = None,
    ):
        # NOTE: This ignores train_Yvar and uses inferred noise instead.
        # squeeze output dim before passing train_Y to ExactGP

        # step 1: build a tree from train_X and train_Y

        # step 1(a): init tree with collected samples and initialise MCTS states
        dims = all_train_X.shape[-1]

        lb = -5 * torch.ones(dims)
        up = 10 * torch.ones(dims)
        ninits = 40
        func = Ackley(dim=dims, negate=True)
        Cp = 1
        leaf_size = 10
        kernel_type = "rbf"
        gamma_type = "auto"

        self.init_tree(
            lb=lb,
            ub=up,
            dims=dims,
            ninits=ninits,
            func=func,
            Cp=Cp,
            leaf_size=leaf_size,
            kernel_type=kernel_type,
            gamma_type=gamma_type,
        )

        # iterate through samples, update curt_best_value etc
        # add to self.samples
        for sample, value in zip(all_train_X, all_train_Y):
            self.collect_samples(sample, value)

        # step 1(b): dynamic_treeify
        self.dynamic_treeify()

        # step 2: select a node and gets its X and fX
        leaf, path = self.select()

        # step 3: initialise a GP on all samples

        super().__init__(all_train_X, all_train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()

        kernel = MaternKernel(nu=2.5, lengthscale_constraint=Interval(1e-5, 1e5))
        kernel.lengthscale = 1.0
        self.covar_module = kernel

        self.to(all_train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def collect_samples(self, sample, value=None):
        # print(sample.shape)
        # exit()
        # TODO: to perform some checks here
        if value == None:
            value = self.func(sample) * -1

        if value > self.curt_best_value:
            self.curt_best_value = value
            self.curt_best_sample = sample
            self.best_value_trace.append((value, self.sample_counter))
        self.sample_counter += 1
        self.samples.append((sample, value))
        return value

    def init_tree(
        self,
        lb: Tensor,
        ub: Tensor,
        dims: int,
        ninits: int,
        func,
        Cp: int = 1,
        leaf_size: int = 20,
        kernel_type="rbf",
        gamma_type="auto",
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

        self.LEAF_SAMPLE_SIZE = leaf_size
        self.kernel_type = kernel_type
        self.gamma_type = gamma_type

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

    def populate_training_data(self):
        # only keep root
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
        self.ROOT.update_bag(self.samples)

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


if __name__ == "__main__":
    ackley = Ackley(dim=10)
    X = torch.rand(20, 10)
    lamcts_gp = LaMCTS_GP()


# need to add constraines to the objective of qEI
