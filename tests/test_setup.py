"""Dummy tests to verify pytest and PyTorch are correctly set up."""

import pytest
import torch

from dynamic import __version__


class TestProjectSetup:
    """Verify the basic project setup is working."""

    def test_version_exists(self):
        """Package version is set."""
        assert __version__ == "0.1.0"

    def test_pytorch_installed(self):
        """PyTorch is importable and reports a version."""
        assert torch.__version__ is not None

    def test_pytorch_version_minimum(self):
        """PyTorch version meets the minimum requirement (>=2.6)."""
        major, minor, *_ = torch.__version__.split(".")
        assert int(major) >= 2
        if int(major) == 2:
            # Strip any suffix like '2.6.0a0+...'
            minor_int = int("".join(c for c in minor if c.isdigit()))
            assert minor_int >= 6


class TestPyTorchBasics:
    """Verify core PyTorch operations work."""

    def test_tensor_creation(self):
        """Can create a basic tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.dtype == torch.float32

    def test_tensor_arithmetic(self):
        """Basic tensor arithmetic works."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = a + b
        assert torch.allclose(result, torch.tensor([5.0, 7.0, 9.0]))

    def test_tensor_matmul(self):
        """Matrix multiplication works."""
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        result = a @ b
        assert result.shape == (3, 5)

    def test_autograd(self):
        """Autograd can compute gradients."""
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2
        y.backward()
        assert torch.allclose(x.grad, torch.tensor([4.0]))

    def test_device_cpu(self):
        """CPU device is always available."""
        device = torch.device("cpu")
        t = torch.zeros(2, 2, device=device)
        assert t.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this machine",
    )
    def test_device_mps(self):
        """MPS device works on Apple Silicon (skipped if unavailable)."""
        device = torch.device("mps")
        t = torch.zeros(2, 2, device=device)
        assert t.device.type == "mps"
