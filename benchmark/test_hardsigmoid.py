import pytest
import torch

from . import base, consts

def hardsigmoid_with_out(input, out=None):
    result = torch.nn.functional.hardsigmoid(input)
    if out is not None:
        out.copy_(result)
        return out
    return result

    
@pytest.mark.hardsigmoid
def test_hardsigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardsigmoid",
        torch_op=torch.nn.functional.hardsigmoid,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


#@pytest.mark.skip(reason="Hardsigmoid doesn't accept 'out': issue #2686")
@pytest.mark.hardsigmoid_out
def test_hardsigmoid_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="hardsigmoid_out",
        torch_op=hardsigmoid_with_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
