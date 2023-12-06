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

import scipy.optimize as sopt
import torch

from openasce.utils.logger import logger


class LBFGSBOptimizer(torch.optim.Optimizer):
    """Wrap L-BFGS-B algorithm, using scipy routines."""

    def __init__(self, params):
        defaults = dict()
        super(LBFGSBOptimizer, self).__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGSBOptimizer doesn't support per-parameter options (parameter groups)"
            )
        self._params = self.param_groups[0]["params"]
        self._numel = sum([p.numel() for p in self._params])

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_bounds(self):
        bounds = []
        for p in self._params:
            if hasattr(p, "bounds"):
                b = p.bounds
            else:
                b = [(None, None)] * p.numel()
            bounds += b
        return bounds

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset : offset + numel].view_as(p.data)
            offset += numel

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """

        def wrapped_closure(flat_params):
            """closure should call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            flat_params = flat_params.to(torch.get_default_dtype())
            self._distribute_flat_params(flat_params)  # update model parameters
            loss = closure()
            loss = loss.item()
            flat_grad = self._gather_flat_grad().cpu().detach().numpy()
            return loss, flat_grad.astype("float64")

        initial_params = self._gather_flat_params()
        initial_params = initial_params.cpu().detach().numpy()
        bounds = self._gather_flat_bounds()
        sol = sopt.minimize(
            wrapped_closure, initial_params, method="L-BFGS-B", jac=True, bounds=bounds
        )
        final_params = torch.from_numpy(sol.x)
        final_params = final_params.to(torch.get_default_dtype())
        self._distribute_flat_params(final_params)
