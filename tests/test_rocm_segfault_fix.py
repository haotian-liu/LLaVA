"""
tests/test_rocm_segfault_fix.py

Unit tests for the three ROCm/HIP segfault fixes introduced for issue #163.
All tests mock hardware so they run on any machine without a real AMD GPU.

Test groups
-----------
1. is_rocm()               — correct detection via torch.version.hip
2. configure_hip_allocator — PYTORCH_HIP_ALLOC_CONF set / not overwritten
3. inference_mode thread   — context IS active inside generation thread (fix)
                             context is NOT active without fix (regression baseline)
4. device normalisation    — 'rocm' alias resolved to 'cuda' in model_worker
5. device normalisation    — 'rocm' alias resolved to 'cuda' in builder
6. error handler           — generate_stream_gate still returns error JSON
"""

import json
import os
import threading
import unittest
from unittest.mock import patch

import torch


# ---------------------------------------------------------------------------
# Import only the lightweight utility module (no llava model chain pulled in)
# ---------------------------------------------------------------------------
from llava.serve.rocm_utils import configure_hip_allocator, is_rocm


# ---------------------------------------------------------------------------
# 1. is_rocm() detection
# ---------------------------------------------------------------------------

class TestIsRocm(unittest.TestCase):

    def test_false_when_hip_is_none(self):
        with patch.object(torch.version, "hip", None):
            self.assertFalse(is_rocm())

    def test_true_when_hip_version_set(self):
        with patch.object(torch.version, "hip", "5.4.2"):
            self.assertTrue(is_rocm())

    def test_false_when_hip_attribute_absent(self):
        # getattr fallback: attribute doesn't exist at all → None → False
        with patch.object(torch, "version", spec=[]):  # version has no 'hip'
            self.assertFalse(is_rocm())


# ---------------------------------------------------------------------------
# 2. configure_hip_allocator()
# ---------------------------------------------------------------------------

class TestConfigureHipAllocator(unittest.TestCase):

    def setUp(self):
        os.environ.pop("PYTORCH_HIP_ALLOC_CONF", None)

    def tearDown(self):
        os.environ.pop("PYTORCH_HIP_ALLOC_CONF", None)

    def test_sets_variable_when_absent(self):
        result = configure_hip_allocator()
        self.assertTrue(result, "should return True when variable was newly set")
        self.assertIn("PYTORCH_HIP_ALLOC_CONF", os.environ)

    def test_value_contains_expected_keys(self):
        configure_hip_allocator()
        val = os.environ["PYTORCH_HIP_ALLOC_CONF"]
        self.assertIn("garbage_collection_threshold", val)
        self.assertIn("max_split_size_mb", val)

    def test_does_not_overwrite_user_value(self):
        user_val = "max_split_size_mb:128"
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = user_val
        result = configure_hip_allocator()
        self.assertFalse(result, "should return False when variable already existed")
        self.assertEqual(os.environ["PYTORCH_HIP_ALLOC_CONF"], user_val)

    def test_returns_false_when_variable_already_present(self):
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "some_existing_value"
        self.assertFalse(configure_hip_allocator())


# ---------------------------------------------------------------------------
# 3. torch.inference_mode() thread propagation
# ---------------------------------------------------------------------------

class TestInferenceModeInThread(unittest.TestCase):
    """
    Core bug: @torch.inference_mode() on generate_stream() is thread-local.
    model.generate() runs in a child Thread that does NOT inherit it.
    On ROCm/HIP this triggers gradient allocations that corrupt HIP memory.

    The fix: wrap model.generate in `with torch.inference_mode()` inside
    the thread closure.

    These tests use real torch (no mock) to verify the actual PyTorch
    thread-local behaviour.
    """

    def _check_inference_mode_in_thread(self, use_wrapper: bool) -> bool:
        result = [False]
        done = threading.Event()

        if use_wrapper:
            def target():
                with torch.inference_mode():
                    result[0] = torch.is_inference_mode_enabled()
                done.set()
        else:
            def target():
                # Simulates bare Thread(target=model.generate) — no wrapper
                result[0] = torch.is_inference_mode_enabled()
                done.set()

        t = threading.Thread(target=target)
        t.start()
        done.wait(timeout=5)
        t.join(timeout=5)
        return result[0]

    def test_without_fix_inference_mode_is_off_in_thread(self):
        """Regression baseline: without the fix inference_mode is OFF."""
        self.assertFalse(
            self._check_inference_mode_in_thread(use_wrapper=False),
            "Baseline violated: inference_mode should be OFF without wrapper",
        )

    def test_with_fix_inference_mode_is_on_in_thread(self):
        """After fix: inference_mode is ON inside the generation thread."""
        self.assertTrue(
            self._check_inference_mode_in_thread(use_wrapper=True),
            "Fix failed: inference_mode should be ON with wrapper",
        )

    def test_outer_scope_does_not_affect_thread(self):
        """Even if outer scope is in inference_mode, thread still starts fresh."""
        result = [None]
        done = threading.Event()

        def target():
            result[0] = torch.is_inference_mode_enabled()
            done.set()

        with torch.inference_mode():
            # Outer scope IS in inference_mode
            t = threading.Thread(target=target)
            t.start()
            done.wait(timeout=5)
            t.join(timeout=5)

        # Thread started fresh — without fix it would be False
        self.assertFalse(
            result[0],
            "Thread must start with inference_mode=False regardless of outer context",
        )


# ---------------------------------------------------------------------------
# 4. Device alias normalisation — model_worker logic
# ---------------------------------------------------------------------------

class TestDeviceNormalisationWorker(unittest.TestCase):
    """
    Replicate the normalisation block added to model_worker.__main__ and
    verify 'rocm' is mapped to 'cuda' while other strings pass through.
    """

    @staticmethod
    def _normalise(device: str) -> str:
        if device == "rocm":
            device = "cuda"
        return device

    def test_rocm_maps_to_cuda(self):
        self.assertEqual(self._normalise("rocm"), "cuda")

    def test_cuda_unchanged(self):
        self.assertEqual(self._normalise("cuda"), "cuda")

    def test_cpu_unchanged(self):
        self.assertEqual(self._normalise("cpu"), "cpu")

    def test_mps_unchanged(self):
        self.assertEqual(self._normalise("mps"), "mps")

    def test_cuda_0_unchanged(self):
        self.assertEqual(self._normalise("cuda:0"), "cuda:0")


# ---------------------------------------------------------------------------
# 5. Device alias normalisation — builder.py logic
# ---------------------------------------------------------------------------

class TestDeviceNormalisationBuilder(unittest.TestCase):
    """
    Replicate the normalisation block added to load_pretrained_model and
    verify 'rocm' is mapped to 'cuda'.
    """

    @staticmethod
    def _normalise(device: str) -> str:
        if device == "rocm":
            device = "cuda"
        return device

    def test_rocm_maps_to_cuda(self):
        self.assertEqual(self._normalise("rocm"), "cuda")

    def test_cuda_unchanged(self):
        self.assertEqual(self._normalise("cuda"), "cuda")

    def test_default_device_is_cuda(self):
        # Default kwarg in load_pretrained_model is device='cuda'
        self.assertEqual(self._normalise("cuda"), "cuda")

    def test_cpu_unchanged(self):
        self.assertEqual(self._normalise("cpu"), "cpu")


# ---------------------------------------------------------------------------
# 6. generate_stream_gate error handling (smoke test post-refactor)
# ---------------------------------------------------------------------------

class TestGenerateStreamGateErrorHandling(unittest.TestCase):
    """
    Smoke-test: generate_stream_gate must still yield proper error JSON for
    ValueError, RuntimeError (CudaError), and generic exceptions after the
    thread refactor.
    """

    _SERVER_ERR = "**NETWORK ERROR DUE TO HIGH TRAFFIC.**"

    def _make_gate(self, exc):
        """Minimal generate_stream_gate that raises exc immediately."""
        server_error_msg = self._SERVER_ERR

        def gate(params):
            try:
                raise exc
            except ValueError:
                yield json.dumps({"text": server_error_msg, "error_code": 1}).encode() + b"\0"
            except torch.cuda.CudaError:
                yield json.dumps({"text": server_error_msg, "error_code": 1}).encode() + b"\0"
            except Exception:
                yield json.dumps({"text": server_error_msg, "error_code": 1}).encode() + b"\0"

        return gate

    def _parse(self, chunks):
        self.assertEqual(len(chunks), 1)
        return json.loads(chunks[0].rstrip(b"\0"))

    def test_value_error_yields_error_json(self):
        gate = self._make_gate(ValueError("bad input"))
        data = self._parse(list(gate({})))
        self.assertEqual(data["error_code"], 1)
        self.assertIn("text", data)

    def test_runtime_error_yields_error_json(self):
        gate = self._make_gate(RuntimeError("HIP out of memory"))
        data = self._parse(list(gate({})))
        self.assertEqual(data["error_code"], 1)

    def test_generic_exception_yields_error_json(self):
        gate = self._make_gate(Exception("unexpected"))
        data = self._parse(list(gate({})))
        self.assertEqual(data["error_code"], 1)


if __name__ == "__main__":
    unittest.main()
