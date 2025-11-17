from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._get_prefix_rep = model.get_prefix_rep
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            # self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)

    @override
    # def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
    def infer(self, obs: dict, *, action_noise: np.ndarray | None = None, cond_t: np.ndarray | None = None) -> dict:  # type: ignore[misc]

        # TODO: for now fitting the naming conventions
        noise = action_noise
        timestep_prefix = cond_t

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        batched = inputs["state"].ndim > 1
        batch_size = inputs["state"].shape[0] if batched else 1

        if not self._is_pytorch_model:
            # Convert leaves to jax.Array and add batch dim if needed.
            def _to_jax_array(x):
                if x is None:
                    return None
                arr = jnp.asarray(x)
                if not batched:
                    arr = arr[np.newaxis, ...]
                return arr

            inputs = jax.tree.map(_to_jax_array, inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            def _to_torch_tensor(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    tensor = x.to(self._pytorch_device)
                else:
                    tensor = torch.from_numpy(np.asarray(x)).to(self._pytorch_device)
                if not batched:
                    tensor = tensor.unsqueeze(0)
                return tensor

            inputs = jax.tree.map(_to_torch_tensor, inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Add batch dim to masks
        if batched:
            for cam in inputs["image"].keys():
                if inputs["image_mask"][cam].ndim == 0:  # scalar
                    m = inputs["image_mask"][cam]
                    if self._is_pytorch_model:
                        inputs["image_mask"][cam] = m.expand(batch_size)
                    else:
                        inputs["image_mask"][cam] = jnp.broadcast_to(m, (batch_size,))

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)

        def _prepare_noise(noise_array: np.ndarray) -> np.ndarray:
            ah = self._model.config.action_horizon
            ad = self._model.config.action_dim
            if isinstance(noise_array, torch.Tensor):
                arr = noise_array.detach().cpu().numpy()
            else:
                arr = np.asarray(noise_array)

            if arr.ndim == 3:
                assert arr.shape[0] == batch_size
            elif arr.ndim == 2:
                if arr.shape == (batch_size, ad):
                    arr = np.repeat(arr[:, None, :], ah, axis=1)
                elif batched and arr.shape == (1, ad):
                    arr = np.repeat(arr, batch_size, axis=0)
                    arr = np.repeat(arr[:, None, :], ah, axis=1)
                else:
                    raise ValueError(f"Unexpected noise shape {arr.shape}")
            else:
                raise ValueError(f"Unsupported noise rank {arr.ndim}")
            return arr.astype(np.float32, copy=False)

        if noise is not None:
            # TODO: fix
            if noise.shape[-1] != self._model.config.action_dim:

                prefix_noise = noise[:, self._model.config.action_dim:]
                noise = noise[:, :self._model.config.action_dim]
                
                prefix_noise = np.repeat(prefix_noise[:, None, :], 816, axis=1) # TODO: hardcoded
                prefix_noise = torch.from_numpy(prefix_noise).to(self._pytorch_device)
                sample_kwargs["noise_prefix"] = prefix_noise

            noise_arr = _prepare_noise(noise)
            if self._is_pytorch_model:
                noise_tensor = torch.from_numpy(noise_arr).to(self._pytorch_device)
            else:
                noise_tensor = jnp.asarray(noise_arr)
            sample_kwargs["noise"] = noise_tensor

        def _prepare_time_prefix(prefix_array: np.ndarray) -> np.ndarray:
            if isinstance(prefix_array, torch.Tensor):
                arr = prefix_array.detach().cpu().numpy()
            else:
                arr = np.asarray(prefix_array)
            if batched:
                if arr.ndim == 0:
                    arr = np.full((batch_size,), arr)
                elif arr.shape[0] == 1 and batch_size > 1:
                    arr = np.repeat(arr, batch_size, axis=0)
                elif arr.shape[0] != batch_size:
                    raise ValueError(f"cond_t batch dim {arr.shape[0]} does not match inputs batch {batch_size}")
                arr = arr.reshape(batch_size, -1)
                if arr.shape[1] != 1:
                    raise ValueError(f"cond_t must provide a single scalar per example, got shape {arr.shape}")
                arr = arr[:, 0]
            else:
                arr = np.reshape(arr, -1)
                if arr.size == 0:
                    raise ValueError("cond_t cannot be empty")
                arr = arr[:1]
            return arr.astype(np.float32, copy=False)

        if timestep_prefix is not None:
            prefix_arr = _prepare_time_prefix(timestep_prefix)
            if self._is_pytorch_model:
                prefix_tensor = torch.from_numpy(prefix_arr).to(self._pytorch_device)
            else:
                prefix_tensor = jnp.asarray(prefix_arr)
            sample_kwargs["time_prefix"] = prefix_tensor


        # if noise_prefix is not None:
        #     noise_prefix_arr = _prepare_noise(noise_prefix)
        #     if self._is_pytorch_model:
        #         noise_prefix_tensor = torch.from_numpy(noise_prefix_arr).to(self._pytorch_device)
        #     else:
        #         noise_prefix_tensor = jnp.asarray(noise_prefix_arr)
        #     sample_kwargs["noise_prefix"] = noise_prefix_tensor

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x.detach().cpu()) if hasattr(x, "detach") else x, outputs
            )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x) if x is not None else None, outputs)

        if not batched:
            outputs = jax.tree.map(lambda x: x[0, ...] if hasattr(x, "ndim") and x.ndim > 0 else x, outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    # @override
    # # def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
    # def infer_batched(self, obs: dict, *, action_noise: np.ndarray | None = None, cond_t: np.ndarray | None = None) -> dict:  # type: ignore[misc]

    #     # TODO: for now fitting the naming conventions
    #     noise = action_noise
    #     timestep_prefix = cond_t

    #     # Make a copy since transformations may modify the inputs in place.
    #     inputs = jax.tree.map(lambda x: x, obs)
    #     inputs = self._input_transform(inputs)
    #     if not self._is_pytorch_model:
    #         # Make a batch and convert to jax.Array.
    #         inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    #         self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
    #     else:
    #         # Convert inputs to PyTorch tensors and move to correct device
    #         inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
    #         sample_rng_or_pytorch_device = self._pytorch_device

    #     # Prepare kwargs for sample_actions
    #     sample_kwargs = dict(self._sample_kwargs)
    #     if noise is not None:
    #         noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

    #         if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
    #             noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
    #         sample_kwargs["noise"] = noise

    #     if timestep_prefix is not None:
    #         timestep_prefix = torch.from_numpy(timestep_prefix).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(timestep_prefix)
    #         sample_kwargs["time_prefix"] = timestep_prefix

    #     observation = _model.Observation.from_dict(inputs)
    #     start_time = time.monotonic()
    #     outputs = {
    #         "state": inputs["state"],
    #         "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
    #     }
    #     model_time = time.monotonic() - start_time
    #     if self._is_pytorch_model:
    #         outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
    #     else:
    #         outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    #     outputs = self._output_transform(outputs)
    #     outputs["policy_timing"] = {
    #         "infer_ms": model_time * 1000,
    #     }
    #     return outputs

    # Reference:
    # https://github.com/nakamotoo/openpi/blob/a6d2400d2534ce32e7bdf8747709b97aaef8ec04/src/openpi/policies/policy.py#L80
    @override
    def get_prefix_rep(self, obs: dict):
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)

        # Add batch dim to masks
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            for cam in inputs["image"].keys():
                if inputs["image_mask"][cam].ndim == 0:  # scalar
                    m = inputs["image_mask"][cam]
                    if self._is_pytorch_model:
                        inputs["image_mask"][cam] = m.expand(batch_size)
                    else:
                        inputs["image_mask"][cam] = jnp.broadcast_to(m, (batch_size,))

        # Add batch dim
        else:
            if self._is_pytorch_model:
                inputs = jax.tree.map(lambda x: x[None, ...], inputs)
            else:
                inputs = jax.tree.map(lambda x: x[np.newaxis, ...], inputs)
        
        return self._get_prefix_rep(_model.Observation.from_dict(inputs))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
