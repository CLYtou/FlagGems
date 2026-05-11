import pytest
import torch

from . import base, consts


def signbit_with_out(input, out=None):
    result = torch.signbit(input)
    if out is not None:
        out.copy_(result)
        return out
    return result


@pytest.mark.signbit
def test_signbit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="signbit", torch_op=torch.signbit, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


#@pytest.mark.skip(reason="No support to non-boolean outputs: issue #2689.")
@pytest.mark.signbit_out
def test_signbit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="signbit_out",
        torch_op=signbit_with_out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
