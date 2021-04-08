from typing import Any, Optional

import numpy as np

import torch
import gpytorch


class FullRankFixedGaussianNoise(gpytorch.module.Module):
    def __init__(self, noise: torch.Tensor) -> None:
        super().__init__()
        self.noise = noise

    def forward(self, *params: Any,
                shape: Optional[torch.Size] = None,
                noise: Optional[torch.Tensor] = None,
                **kwargs: Any
                ) -> gpytorch.lazy.LazyTensor:
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]

        if noise is not None:
            return gpytorch.lazy.NonLazyTensor(noise)
        elif shape[-1] == self.noise.shape[-1]:
            return gpytorch.lazy.NonLazyTensor(self.noise)
        else:
            return gpytorch.lazy.ZeroLazyTensor()

    def _apply(self, fn):
        self.noise = fn(self.noise)
        return super(FullRankFixedGaussianNoise, self)._apply(fn)


class MinimalFixedNoiseMultiTaskGaussianLikelihood(
        gpytorch.likelihoods._GaussianLikelihoodBase):
    def __init__(self, num_tasks, noise_covar, batch_shape=torch.Size()):
        """
        Args:
            num_tasks (int):
                Number of tasks.
            noise_covar (:obj:`gpytorch.module.Module`):
                A model for the noise covariance.
            batch_shape (torch.Size):
                Number of batches.
        """
        super().__init__(noise_covar=noise_covar)
        self.num_tasks = num_tasks

    def marginal(self, function_dist, *params, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        noise_covar = self.noise_covar(*params, shape=covar.shape, **kwargs)
        covar = covar + noise_covar

        return function_dist.__class__(mean, covar)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, rank):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        n_task = train_y.shape[1]

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=n_task
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=n_task, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,
                                                                  covar_x)


def reorder_matrix(M, n):
    """Reorder Kroenecker-product matrix"""
    P = np.empty_like(M)
    m = M.shape[0]//n
    for i in range(n):
        for j in range(n):
            P[i*m:(i+1)*m, j*m:(j+1)*m] = M[i::n, j::n]
    return P


class CIBModel:
    def __init__(self, X, Y, Y_cov):
        self.n_task = Y.shape[1]
        self.task_covar_rank = 1

        self.train_x = torch.tensor(X, dtype=torch.float32)
        self.train_y = torch.tensor(Y, dtype=torch.float32)
        self.train_y_cov = torch.tensor(reorder_matrix(Y_cov, Y.shape[0]),
                                        dtype=torch.float32)

        self._create_model_and_likelihood()

    def print_model_parameters(self):
        print("Model parameters")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(" ", name, param.data, flush=True)

    def _create_model_and_likelihood(self):
        self.likelihood = MinimalFixedNoiseMultiTaskGaussianLikelihood(
                        noise_covar=FullRankFixedGaussianNoise(
                                                    noise=self.train_y_cov),
                        num_tasks=self.n_task)
        self.model = MultitaskGPModel(self.train_x, self.train_y,
                                      self.likelihood,
                                      rank=self.task_covar_rank)

    def load_state(self, state_file=None, state_dict=None):
        if state_dict is None:
            state_dict = torch.load(state_file)

        self._create_model_and_likelihood()
        self.model.load_state_dict(state_dict)

    def save_state(self, state_file):
        torch.save(self.model.state_dict(), state_file)

    def train(self, n_step=100, progress_bar=True):
        training_iterations = n_step

        self.model.train()
        self.likelihood.train()

        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.model)

        if progress_bar:
            import tqdm
            progress = tqdm.trange(training_iterations)
        else:
            progress = range(training_iterations)

        for i in progress:
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if progress_bar:
                progress.set_postfix(loss=loss.item())
            optimizer.step()

    def chi2(self):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            mean = self.model(self.train_x).mean
            mean = mean.view(*mean.shape[:-2], -1)
            cov = self.train_y_cov

            diff = mean - self.train_y.view(*self.train_y.shape[:-2], -1)
            chi2 = diff @ cov.inverse() @ diff

        return chi2.item()

    def predict(self, test_x, mean=True, sample=False, CI=False):
        self.model.eval()
        self.likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor(test_x, dtype=torch.float32)
            f_predictions = self.model(test_x)

        result = []
        if mean:
            result.append(f_predictions.mean.numpy())
        if sample:
            result.append(f_predictions.rsample().numpy())
        if CI:
            l, u = f_predictions.confidence_region()
            result.append((l.numpy(), u.numpy()))

        if len(result) == 1:
            return result[0]
        else:
            return tuple(result)
