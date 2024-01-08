# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import json
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import norm
import copy as cp
from sklearn.svm import SVC

from torch.quasirandom import SobolEngine
from mpl_toolkits.mplot3d import axes3d, Axes3D

from botorch.acquisition.objective import ConstrainedMCObjective
from sklearn.gaussian_process import GaussianProcessRegressor
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import MaternKernel
from gpytorch.constraints import Interval
import gpytorch
from botorch import fit_gpytorch_model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import qExpectedImprovement

from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import matplotlib.pyplot as plt
from matplotlib import cm
import os

# from turbo_1.turbo_1 import Turbo1


# the input will be samples!
class Classifier:
    def __init__(
        self, samples, dims, kernel_type, gamma_type="auto", use_botorch=False
    ):
        self.training_counter = 0
        assert dims >= 1
        assert type(samples) == type([])
        self.dims = dims

        self.use_botorch = use_botorch
        if not self.use_botorch:
            # # create a gaussian process regressor
            noise = 0.1
            m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            self.gpr = GaussianProcessRegressor(
                kernel=m52, alpha=noise**2
            )  # default to CPU

        self.kmean = KMeans(n_clusters=2, n_init="auto")
        # learned boundary
        self.svm = SVC(kernel=kernel_type, gamma=gamma_type)
        # data structures to store
        self.samples = []
        self.X = np.array([])
        self.fX = np.array([])

        # good region is labeled as zero
        # bad  region is labeled as one
        self.good_label_mean = -1
        self.bad_label_mean = -1

        self.update_samples(samples)

    def fit_gpr(self, X, fX):
        X_tensor = torch.from_numpy(X)
        fX_tensor = torch.from_numpy(fX).unsqueeze(-1)
        print(f"X_tensor: {X_tensor.shape}")
        print(f"fX_tensor: {fX_tensor.shape}")
        noise = 0.2
        fX_var = torch.full_like(fX_tensor, noise**2)

        # fX_var = noise * torch.full_like(fX_tensor, noise)

        kernel = MaternKernel(nu=2.5, lengthscale_constraint=Interval(1e-5, 1e5))
        kernel.lengthscale = 1.0
        model = SingleTaskGP(X_tensor, fX_tensor, fX_var, covar_module=kernel)
        # model = SingleTaskGP(X_tensor, fX_tensor, fX_var, covar_module=kernel)
        # model = ExactGPModel(X_tensor, fX_tensor, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_model(mll)
        # training_iter = 2000
        # for i in range(training_iter):
        #     optimizer.zero_grad()
        #     output = model(X_tensor)
        #     loss = -mll(output, fX_tensor)
        #     loss.backward()
        #     print(
        #         f"Iter {i+1}/{training_iter} - Loss: {loss.item()}  lengthscale: {model.covar_module.lengthscale.item()}   noise: {model.likelihood.noise.mean()}"
        #     )
        #     optimizer.step()
        # print(model.covar_module.lengthscale.item())
        self.gpr = model

    def is_splittable_svm(self):
        plabel = self.learn_clusters()
        if len(np.unique(plabel)) == 1:
            return False
        self.learn_boundary(plabel)
        svm_label = self.svm.predict(self.X)
        if len(np.unique(svm_label)) == 1:
            return False
        else:
            return True

    def get_max(self):
        return np.max(self.fX)

    def plot_samples_and_boundary(self, func, name):
        assert func.dims == 2

        plabels = self.svm.predict(self.X)
        good_counts = len(self.X[np.where(plabels == 0)])
        bad_counts = len(self.X[np.where(plabels == 1)])
        good_mean = np.mean(self.fX[np.where(plabels == 0)])
        bad_mean = np.mean(self.fX[np.where(plabels == 1)])

        if np.isnan(good_mean) == False and np.isnan(bad_mean) == False:
            assert good_mean > bad_mean

        lb = func.lb
        ub = func.ub
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        xv, yv = np.meshgrid(x, y)
        true_y = []
        for row in range(0, xv.shape[0]):
            for col in range(0, xv.shape[1]):
                x = xv[row][col]
                y = yv[row][col]
                true_y.append(func(np.array([x, y])))
        true_y = np.array(true_y)
        pred_labels = self.svm.predict(np.c_[xv.ravel(), yv.ravel()])
        pred_labels = pred_labels.reshape(xv.shape)

        fig, ax = plt.subplots()
        ax.contour(xv, yv, true_y.reshape(xv.shape), cmap=cm.coolwarm)
        ax.contourf(xv, yv, pred_labels, alpha=0.4)

        ax.scatter(
            self.X[np.where(plabels == 0), 0],
            self.X[np.where(plabels == 0), 1],
            marker="x",
            label="good-" + str(np.round(good_mean, 2)) + "-" + str(good_counts),
        )
        ax.scatter(
            self.X[np.where(plabels == 1), 0],
            self.X[np.where(plabels == 1), 1],
            marker="x",
            label="bad-" + str(np.round(bad_mean, 2)) + "-" + str(bad_counts),
        )
        ax.legend(loc="best")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig(name)
        plt.close()

    def get_mean(self):
        return np.mean(self.fX)

    def update_samples(self, latest_samples):
        assert type(latest_samples) == type([])
        X = []
        fX = []
        for sample in latest_samples:
            X.append(sample[0])
            fX.append(sample[1])

        self.X = np.asarray(X, dtype=np.float32).reshape(-1, self.dims)
        self.fX = np.asarray(fX, dtype=np.float32).reshape(-1)
        self.samples = latest_samples

    def train_gpr(self, samples):
        X = []
        fX = []
        for sample in samples:
            X.append(sample[0])
            fX.append(sample[1])
        X = np.asarray(X).reshape(-1, self.dims)
        fX = np.asarray(fX).reshape(-1)

        if self.use_botorch:
            self.fit_gpr(X, fX)
        else:
            self.gpr.fit(X, fX)

    ###########################
    # BO sampling with EI
    ###########################

    def expected_improvement(self, X, xi=0.0001, use_ei=True):
        """Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
        Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
        Returns: Expected improvements at points X."""
        X_sample = self.X
        Y_sample = self.fX.reshape((-1, 1))
        X_sample_tensor = torch.from_numpy(X_sample)

        gpr = self.gpr

        if self.use_botorch:
            gpr.eval()
            X_tensor = torch.from_numpy(X)
            prediction = gpr(X_tensor)
            mu = prediction.mean.detach().numpy()
            print(f"mu: {mu.shape}, {type(mu)}")
            sigma = prediction.variance.detach().numpy()
        else:
            mu, sigma = gpr.predict(X, return_std=True)
            print(mu.shape)

        if not use_ei:
            return mu
        else:
            # calculate EI
            if self.use_botorch:
                mu_sample = gpr.forward(X_sample_tensor).mean.detach().numpy()
            else:
                mu_sample = gpr.predict(X_sample)
            sigma = sigma.reshape(-1, 1)
            mu_sample_opt = np.max(mu_sample)
            with np.errstate(divide="warn"):
                imp = mu - mu_sample_opt - xi
                imp = imp.reshape((-1, 1))
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei

    def plot_boundary(self, X):
        if X.shape[1] > 2:
            return
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], marker=".")
        ax.scatter(self.X[:, 0], self.X[:, 1], marker="x")
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig("boundary.pdf")
        plt.close()

    def get_sample_ratio_in_region(self, cands, path):
        total = len(cands)
        for node in path:
            boundary = node[0].classifier.svm
            if len(cands) == 0:
                return 0, np.array([])
            assert len(cands) > 0
            cands = cands[boundary.predict(cands) == node[1]]
            # node[1] store the direction to go
        ratio = len(cands) / total
        assert len(cands) <= total
        return ratio, cands

    def propose_rand_samples_sobol(self, nums_random_samples, path, lb, ub):
        # rejected sampling
        selected_cands = np.zeros((1, self.dims))
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)

        # scale the samples to the entire search space
        # ----------------------------------- #
        # while len(selected_cands) <= nums_samples:
        #     cands  = sobol.draw(100000).to(dtype=torch.float64).cpu().detach().numpy()
        #     cands  = (ub - lb)*cands + lb
        #     for node in path:
        #         boundary = node[0].classifier.svm
        #         if len(cands) == 0:
        #             return []
        #         cands = cands[ boundary.predict(cands) == node[1] ] # node[1] store the direction to go
        #     selected_cands = np.append( selected_cands, cands, axis= 0)
        #     print("total sampled:", len(selected_cands) )
        # return cands
        # ----------------------------------- #
        # shrink the cands region

        ratio_check, centers = self.get_sample_ratio_in_region(self.X, path)
        # no current samples located in the region
        # should not happen
        # print("ratio check:", ratio_check, len(self.X) )
        # assert ratio_check > 0
        if ratio_check == 0 or len(centers) == 0:
            return self.propose_rand_samples(nums_random_samples, lb, ub)

        lb_ = None
        ub_ = None

        final_cands = []
        for center in centers:
            center = self.X[np.random.randint(len(self.X))]
            cands = sobol.draw(2000).to(dtype=torch.float64).cpu().detach().numpy()
            ratio = 1
            L = 0.0001
            Blimit = np.max(ub - lb)

            while ratio == 1 and L < Blimit:
                lb_ = np.clip(center - L / 2, lb, ub)
                ub_ = np.clip(center + L / 2, lb, ub)
                cands_ = cp.deepcopy(cands)
                cands_ = (ub_ - lb_) * cands_ + lb_
                ratio, cands_ = self.get_sample_ratio_in_region(cands_, path)
                if ratio < 1:
                    final_cands.extend(cands_.tolist())
                L = L * 2
        final_cands = np.array(final_cands)
        if len(final_cands) > nums_random_samples:
            final_cands_idx = np.random.choice(len(final_cands), nums_random_samples)
            return final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                return self.propose_rand_samples(nums_random_samples, lb, ub)
            else:
                return final_cands

    def propose_samples_botorch(
        self, nums_samples=10, path=None, lb=None, ub=None, samples=None, bags=None
    ):
        """Proposes the next sampling point by optimizing the acquisition function.
        Args: acquisition: Acquisition function. X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples.
        Returns: Location of the acquisition function maximum."""
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0

        self.train_gpr(samples)  # learn in unit cube

        dim = self.dims
        if len(path) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)

        MC_SAMPLES = 1000
        BATCH_SIZE = nums_samples
        NUM_RESTARTS = 10
        RAW_SAMPLES = 1000
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        qEI = qExpectedImprovement(
            model=self.gpr,
            best_f=self.get_max(),  # need to multiply by constraint
            sampler=qmc_sampler,
        )

        constraints = []
        for node in path:
            print(node[0].get_name())
            boundary = node[0].classifier.svm

            def constraint(X):
                # print(f"In constraint X at {node[0].get_name()}")
                if (
                    torch.from_numpy(
                        boundary.predict(X.detach().numpy().reshape(1, -1))
                    )
                    == node[1]
                ):
                    assert torch.norm(X) >= 0
                    return torch.norm(X)
                else:
                    assert -1 * torch.norm(X) < 0
                    return -1 * torch.norm(X)

            constraints.append((constraint, True))

        # for x in bags:
        #     print(f"Checking if all the conditions are valid")
        #     for constraint in constraints:
        #         if not constraint[0](torch.tensor(x[0])) >= 0:
        #             return None

        bounds = torch.stack([torch.from_numpy(lb), torch.from_numpy(ub)], dim=0)

        def ic_generator(
            acq_function,
            bounds,
            q,
            num_restarts,
            raw_samples,
            fixed_features=None,
            options=None,
            inequality_constraints=None,
            equality_constraints=None,
            generator=None,
            fixed_X_fantasies=None,
        ):
            # nums_rand_samples = num_restarts * q
            # Xinit = self.propose_rand_samples_sobol_reject(
            #     nums_rand_samples, path, lb, ub
            # )

            # print(f"Xinit with reject sampling: {Xinit.shape}")
            # index = np.arange(len(Xinit))
            # Xinit_index = np.random.choice(index, num_restarts * q, replace=True)
            # Xinit = Xinit[Xinit_index]

            # Xinit = torch.tensor(Xinit).reshape(num_restarts, q, dim)
            # print(f"Xinit: {Xinit.shape}")
            # return Xinit

            Xinit = self.propose_rand_samples_sobol(
                nums_random_samples=num_restarts * q * 1000, path=path, lb=lb, ub=ub
            )
            valid_X = []
            for x in Xinit:
                is_in_region = True
                for constraint in constraints:
                    if constraint[0](torch.tensor(x)) < 0:
                        is_in_region = False
                        break
                if is_in_region:
                    valid_X.append(x)

                    for constraint in constraints:
                        assert constraint[0](torch.tensor(x)) >= 0

            if len(valid_X) == 0:
                print("No valid X found, start from existing samples")
                print(f"Number of elements in the current node: {len(bags)}")
                for x in bags:
                    print(x)

                    for constraint in constraints:
                        if not constraint[0](torch.tensor(x[0])) >= 0:
                            raise ValueError
                        # assert (
                        #     constraint[0](torch.tensor(x[0])) >= 0
                        # ), "elements in bags must satisfy the constraint"
                    valid_X.append(x[0])
                # for x in self.X:
                #     is_in_region = True
                #     for constraint in constraints:
                #         if constraint[0](torch.tensor(x)) < 0:
                #             is_in_region = False
                #             break
                #     if is_in_region:
                #         valid_X.append(x)

            valid_X = np.array(valid_X)
            print(f"valid_X: {valid_X.shape}")

            index = np.arange(len(valid_X))
            Xinit_index = np.random.choice(index, num_restarts * q, replace=True)
            Xinit = valid_X[Xinit_index]

            for xinit in Xinit:
                for constraint in constraints:
                    assert constraint[0](torch.tensor(xinit)) >= 0

            Xinit = torch.tensor(Xinit).reshape(num_restarts, q, dim)
            print(f"Xinit: {Xinit.shape}")

            return Xinit

            # valid_X = []
            # for x in self.X:
            #     is_in_region = True
            #     for node in path:
            #         boundary = node[0].classifier.svm
            #         if boundary.predict(x.reshape(1, -1)) != node[1]:
            #             is_in_region = False
            #             break
            #     if is_in_region:
            #         valid_X.append(x)
            #     valid_X.append(x)
            # valid_X = np.array(valid_X)
            # index = np.arange(len(valid_X))

            # print(f"valid_X: {len(valid_X)}")
            # Xinit_index = np.random.choice(index, num_restarts * q, replace=True)
            # Xinit = valid_X[Xinit_index]
            # print("Xinit: ", Xinit)
            # Xinit = torch.tensor(Xinit)
            # Xinit.requires_grad = False
            # Xinit = Xinit.reshape(num_restarts, q, dim)
            # print(f"Xinit: {Xinit.shape}, {type(Xinit)}, {Xinit.requires_grad}")
            # return Xinit

        # Xinit = gen_batch_initial_conditions(
        #     acq_function=qEI,
        #     bounds=bounds,
        #     q=BATCH_SIZE,
        #     num_restarts=NUM_RESTARTS,
        #     raw_samples=RAW_SAMPLES,
        # )
        # print(f"Xinit: {Xinit.shape}")

        candidates, _ = optimize_acqf(
            acq_function=qEI,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 1, "maxiter": 50},
            nonlinear_inequality_constraints=constraints,
            ic_generator=ic_generator,
        )

        print(f"candicates: {candidates.shape}")
        return candidates.detach().numpy()

        # if len(X) == 0:
        #     return self.propose_rand_samples(nums_samples, lb, ub)

        # X_ei = self.expected_improvement(X, xi=0.001, use_ei=True)
        # row, col = X.shape

        # X_ei = X_ei.reshape(len(X))
        # n = nums_samples
        # if X_ei.shape[0] < n:
        #     n = X_ei.shape[0]
        # indices = np.argsort(X_ei)[-n:]
        # proposed_X = X[indices]
        # return proposed_X

    def propose_samples_bo(
        self, nums_samples=10, path=None, lb=None, ub=None, samples=None
    ):
        """Proposes the next sampling point by optimizing the acquisition function.
        Args: acquisition: Acquisition function. X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples.
        Returns: Location of the acquisition function maximum."""
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0

        self.train_gpr(samples)  # learn in unit cube

        dim = self.dims
        nums_rand_samples = 10000
        if len(path) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)

        # def obj_callable(X):
        #     print(X.shape)
        #     return X

        # def constraint_callable(X):
        #     # X is of shape sample_shape * batch_shape * q * o
        #     # ans = 0
        #     # for node in path:
        #     #     boundary = node[0].classifier.svm
        #     #     label = boundary.predict(X)
        #     #     if np.any(label != node[1]):
        #     #         ans = 1
        #     #         break
        #     return 0  # always feasible

        # constrained_obj = ConstrainedMCObjective(
        #     objective=obj_callable, constraints=[constraint_callable]
        # )

        # MC_SAMPLES = 1000
        # BATCH_SIZE = 1
        # NUM_RESTARTS = 1
        # RAW_SAMPLES = 100
        # qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        # qEI = qExpectedImprovement(
        #     model=self.gpr,
        #     best_f=self.get_max(),  # need to multiply by constraint
        #     sampler=qmc_sampler,
        # )
        # test_x = torch.rand(20, 10)
        # samples = qmc_sampler(self.gpr.posterior(test_x))
        # print(f"Samples shape: {samples.shape}")
        # objective = constrained_obj(samples)
        # print(f"Objective shape: {objective.shape}")

        # bounds = torch.stack([torch.from_numpy(lb), torch.from_numpy(ub)], dim=0)
        # print(bounds.shape)
        # candicates, values = optimize_acqf(
        #     acq_function=qEI,
        #     bounds=bounds,
        #     q=BATCH_SIZE,
        #     num_restarts=NUM_RESTARTS,
        #     raw_samples=RAW_SAMPLES,
        #     options={"batch_limit": 5, "maxiter": 200},
        # )
        # print(f"candicates: {candicates.shape}")
        # X = candicates.detach().numpy()

        X = self.propose_rand_samples_sobol(nums_rand_samples, path, lb, ub)
        # print(f"sobol_proposed_X: {X.shape}")

        # print(path)
        # for node in path:
        #     boundary = node[0].classifier.svm
        #     print(node[0].get_name(), node[1])
        #     label = boundary.predict(X)
        #     assert np.all(
        #         label == node[1]
        #     ), f"label not consistent at node {node[0].get_name()}"

        # label = self.svm.predict(X)

        # print(f"X: {X.shape}")
        # print(f"X[0]: {X[0:1].shape}")

        # label = self.svm.predict(X)
        # print(f"label of X: {type(label)}, {label.shape}")
        # print(f"{np.sum(label==0)} good samples")

        # print("samples in the region:", len(X) )
        # self.plot_boundary(X)
        if len(X) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)

        X_ei = self.expected_improvement(X, xi=0.001, use_ei=True)
        row, col = X.shape

        X_ei = X_ei.reshape(len(X))
        n = nums_samples
        if X_ei.shape[0] < n:
            n = X_ei.shape[0]
        indices = np.argsort(X_ei)[-n:]
        proposed_X = X[indices]
        return proposed_X

    ###########################
    # random sampling
    ###########################

    def propose_rand_samples(self, nums_samples, lb, ub):
        x = np.random.uniform(lb, ub, size=(nums_samples, self.dims))
        return x

    def propose_samples_rand(self, nums_samples=10):
        return self.propose_rand_samples(nums_samples, self.lb, self.ub)

    ###########################
    # learning boundary
    ###########################

    def get_cluster_mean(self, plabel):
        assert plabel.shape[0] == self.fX.shape[0]

        zero_label_fX = []
        one_label_fX = []

        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                zero_label_fX.append(self.fX[idx])
            elif plabel[idx] == 1:
                one_label_fX.append(self.fX[idx])
            else:
                print("kmean should only predict two clusters, Classifiers.py:line73")
                os._exit(1)

        good_label_mean = np.mean(np.array(zero_label_fX))
        bad_label_mean = np.mean(np.array(one_label_fX))
        return good_label_mean, bad_label_mean

    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.X)
        self.svm.fit(self.X, plabel)

    def learn_clusters(self):
        assert len(self.samples) >= 2, "samples must > 0"
        assert self.X.shape[0], "points must > 0"
        assert self.fX.shape[0], "fX must > 0"
        assert self.X.shape[0] == self.fX.shape[0]

        tmp = np.concatenate((self.X, self.fX.reshape([-1, 1])), axis=1)
        assert tmp.shape[0] == self.fX.shape[0]

        self.kmean = self.kmean.fit(tmp)
        plabel = self.kmean.predict(tmp)

        # the 0-1 labels in kmean can be different from the actual
        # flip the label if not consistent
        # 0: good cluster, 1: bad cluster

        self.good_label_mean, self.bad_label_mean = self.get_cluster_mean(plabel)

        if self.bad_label_mean > self.good_label_mean:
            for idx in range(0, len(plabel)):
                if plabel[idx] == 0:
                    plabel[idx] = 1
                else:
                    plabel[idx] = 0

        self.good_label_mean, self.bad_label_mean = self.get_cluster_mean(plabel)

        return plabel

    def split_data(self):
        good_samples = []
        bad_samples = []
        train_good_samples = []
        train_bad_samples = []
        if len(self.samples) == 0:
            return good_samples, bad_samples

        plabel = self.learn_clusters()
        self.learn_boundary(plabel)

        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                # ensure the consistency
                assert self.samples[idx][-1] - self.fX[idx] <= 1
                good_samples.append(self.samples[idx])
                train_good_samples.append(self.X[idx])
            else:
                bad_samples.append(self.samples[idx])
                train_bad_samples.append(self.X[idx])

        train_good_samples = np.array(train_good_samples)
        train_bad_samples = np.array(train_bad_samples)

        assert len(good_samples) + len(bad_samples) == len(self.samples)

        return good_samples, bad_samples
