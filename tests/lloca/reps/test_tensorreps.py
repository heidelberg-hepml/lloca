from lloca.reps.tensorreps import TensorReps


def test_is_simplified_distinguishes_parity():
    """Reps of the same order but opposite parity are distinct and must not be merged."""
    assert TensorReps("2x0n+3x1n").is_simplified()
    assert TensorReps("2x1n+3x1p", simplify=False).is_simplified()
    assert not TensorReps("2x1n+3x1n", simplify=False).is_simplified()
    assert not TensorReps("3x1n+2x0n", simplify=False).is_simplified()  # not sorted

    assert TensorReps("2x1n+3x1n", simplify=False).simplify() == TensorReps("5x1n")
