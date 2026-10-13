from types import SimpleNamespace

from profiling.gpu_info import set_matmul_precision


def test_new_precision_api_does_not_write_legacy_tf32_flag():
    backend = SimpleNamespace(fp32_precision="ieee", allow_tf32=False)
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend)))
    assert set_matmul_precision(torch, "tf32").endswith("fp32_precision")
    assert backend.fp32_precision == "tf32"
    assert backend.allow_tf32 is False


def test_legacy_assertion_proxy_uses_compatible_tf32_setting():
    class Backend:
        allow_tf32 = True

        def __getattr__(self, name):
            raise AssertionError(f"Unknown attribute {name}")

    backend = Backend()
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend)))
    assert "legacy" in set_matmul_precision(torch, "ieee")
    assert backend.allow_tf32 is False
