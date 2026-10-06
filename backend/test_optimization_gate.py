from app.services.optimization_gate import OptimizationGate


def test_optimization_gate_direct_task_default_mode():
    gate_res = OptimizationGate.evaluate("Convert this CSV to JSON.")
    assert gate_res["should_optimize"] is False
    assert gate_res["is_ambiguous"] is False


def test_optimization_gate_coordinate_action_bypasses():
    gate_res = OptimizationGate.evaluate(
        "make a python script to read csv and remove duplicates",
        mode="code",
    )
    assert gate_res["should_optimize"] is False


def test_optimization_gate_multi_clause_with_then_does_not_bypass():
    gate_res = OptimizationGate.evaluate(
        "make a python script to read csv and then remove duplicates",
        mode="code",
    )
    assert gate_res["should_optimize"] is True


def test_optimization_gate_multiple_and_conjunctions_does_not_bypass():
    gate_res = OptimizationGate.evaluate(
        "make a script to read csv and remove duplicates and send email",
        mode="code",
    )
    assert gate_res["should_optimize"] is True


def test_optimization_gate_preserves_ambiguity_block():
    gate_res = OptimizationGate.evaluate("write code")
    assert gate_res["is_ambiguous"] is True
    assert gate_res["should_optimize"] is False
