#    Copyright 2023 AntGroup CO., Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# Some of the code implementation is referred from https://github.com/xunzheng/notears
# Modified by Ant Group in 2023

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from openasce.discovery.regression_discovery.lbfgsb_optimizer import LBFGSBOptimizer
from openasce.utils.logger import logger


class Strategy(object):
    """General class to implement different structure learning methods

    Attributes

    """

    def __init__(self, node_names: List[str], **kwargs):
        """Contructor

        Arguments:
            node_names: the name of nodes
        """
        self.node_names = node_names

    def run(
        self,
        *,
        model: nn.Module,
        data: np.ndarray,
        max_iteration: int = 3,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        **kwargs,
    ) -> Tuple:
        """Run the actual strategy

        Arguments:
            model: the model used to discover the better graph
            data: the features of samples
            **kwargs (dict): dictionnary with method specific args

        Returns:

        """
        rho, alpha, h = 1.0, 0.0, np.inf
        for curr_step in range(max_iteration):
            try:
                rho, alpha, h = self.dual_ascent(
                    model, data, lambda1, lambda2, rho_max, rho, alpha, h
                )
            except Exception as e:
                logger.info(f"Exception happens. Current step={curr_step}:\n{e}")
                raise
            finally:
                logger.info(f"Finish step={curr_step}, rho={rho}, alpha={alpha}, h={h}")
            if h <= h_tol or rho >= rho_max:
                break
        W_est = model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est

    def dual_ascent(self, model, X, lambda1, lambda2, rho_max, rho, alpha, h):
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        optimizer = LBFGSBOptimizer(model.parameters())
        X_torch = torch.from_numpy(X)
        rho_times = 0
        while rho < rho_max:
            logger.info(f"rho_times={rho_times}")

            def closure():
                optimizer.zero_grad()
                X_hat = model(X_torch)
                loss = squared_loss(X_hat, X_torch)
                h_val = model.h_func()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * model.l2_reg()
                l1_reg = lambda1 * model.fc1_l1_reg()
                primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj

            optimizer.step(closure)
            with torch.no_grad():
                h_new = model.h_func().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
            rho_times += 1
        alpha += rho * h_new
        return rho, alpha, h_new


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss
