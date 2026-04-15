from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_train_mlp_fermi_objective_demo_runs():
    path = Path(__file__).resolve().parents[1] / "examples" / "train_mlp_fermi_objective.py"
    spec = spec_from_file_location("train_mlp_fermi_objective", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    run_demo = module.run_demo
    _model, tracker, report = run_demo(epochs=2, batch_size=128, seed=11)

    assert len(tracker.records) >= 2
    assert report.total_weights > 0
    assert 0.0 <= report.fraction_kept <= 1.0
    assert 0.0 <= report.saved_macs_fraction <= 1.0
