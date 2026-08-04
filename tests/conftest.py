"""Shared pytest fixtures for gtrack tests."""

import functools
import importlib.util
import inspect
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
_MAKER = DATA_DIR / "make_synthetic_model.py"


def signature_bound(real, fake):
    """Wrap ``fake`` so it only accepts calls ``real`` would also accept.

    A fake written as ``def fake(a, b, **kwargs)`` accepts every keyword there
    has ever been, including ones the real callable does not take. The test
    then passes while production raises ``TypeError`` on the same call, which
    is not a hypothetical: ``PolygonIndicatorSource`` was written against a
    ``build_indicator_source`` that had gained ``membership_property``, its
    suite was fully green, and the parameter existed in no committed version of
    the function.

    That failure mode is worth spending a wrapper on because of where these
    sources are consumed. Under MPI the call happens on rank 0 inside a
    collective, so a ``TypeError`` there is not a traceback — it kills rank 0
    while every other rank blocks in ``bcast``, and the job hangs instead of
    failing.

    Binding against the real signature costs nothing at runtime and turns that
    drift into an ordinary test failure at the point the fake is called.

    Args:
        real: the callable being stood in for. Its signature is the contract.
              For a class, pass the class; its ``__init__`` is used.
        fake: the stand-in actually invoked once the arguments bind.

    Returns:
        A callable delegating to ``fake``, raising ``TypeError`` first if the
        arguments would not bind to ``real``.
    """
    if inspect.isclass(real):
        return _bound_class(real, fake)
    return _bound_call(real, fake, inspect.signature(real), real.__name__)


def _bound_call(real, fake, sig, label):
    """Return ``fake`` guarded by ``sig``."""

    @functools.wraps(fake)
    def bound(*args, **kwargs):
        try:
            sig.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"call does not match the real {label}{sig}: {exc}. "
                "The fake accepted it, production would not."
            ) from exc
        return fake(*args, **kwargs)

    return bound


def _drop_self(sig):
    return sig.replace(parameters=list(sig.parameters.values())[1:])


def _bound_class(real, fake):
    """Guard a fake class's constructor and every method the real class shares.

    Methods matter as much as the constructor here. A fake's ``rotate`` or
    ``filter_inside`` is just as free to accept an argument list the real one
    would reject, and the source recipes call those methods far more often than
    they construct anything.
    """
    init_sig = _drop_self(inspect.signature(real.__init__))

    @functools.wraps(fake, updated=())
    def construct(*args, **kwargs):
        try:
            init_sig.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"construction does not match the real "
                f"{real.__name__}{init_sig}: {exc}. "
                "The fake accepted it, production would not."
            ) from exc
        instance = fake(*args, **kwargs)
        for name in dir(instance):
            if name.startswith("_"):
                continue
            fake_attr = getattr(instance, name)
            real_attr = getattr(real, name, None)
            if not callable(fake_attr) or not callable(real_attr):
                continue
            label = f"{real.__name__}.{name}"
            sig = _drop_self(inspect.signature(real_attr))
            setattr(instance, name, _bound_call(real_attr, fake_attr, sig, label))
        return instance

    return construct


@pytest.fixture
def bind_signature():
    """Expose :func:`signature_bound` to test modules.

    A fixture rather than a plain import because ``tests`` is not a package and
    pytest loads ``conftest`` under its own plugin name, so a test module
    cannot ``from conftest import`` it.
    """
    return signature_bound


def _ensure_synthetic_model():
    """Generate the synthetic topological model fixtures if missing."""
    spec = importlib.util.spec_from_file_location("_make_synthetic_model", _MAKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paths = (mod.ROT_PATH, mod.TOPO_PATH, mod.STATIC_PATH)
    if not all(p.exists() for p in paths):
        mod.main()
    return mod


@pytest.fixture(scope="session")
def synthetic_model():
    """Paths and key facts for the tiny synthetic topological model.

    See tests/data/make_synthetic_model.py for how it is constructed. Returns a
    dict with the file paths and the geometry/plate facts the tests rely on.
    """
    mod = _ensure_synthetic_model()
    return {
        "rotation_files": str(mod.ROT_PATH),
        "topology_files": str(mod.TOPO_PATH),
        "static_polygons": str(mod.STATIC_PATH),
        # A point inside plate A's rigid topological boundary (moves +30 deg).
        "plate_a_point": (10.0, 10.0),          # (lat, lon)
        # A point inside the deforming network (deforms, not a clean rigid pole).
        "network_point": (0.0, 90.0),
        # A point in a topology gap (kept, unmoved with deactivate_points=None).
        "gap_point": (0.0, -170.0),
        # The out-of-circuit point: inside plate A topologically, but the static
        # polygon labels it plate 999 which has no rigid sequence.
        "out_of_circuit_point": mod.OUT_OF_CIRCUIT_POINT,
        "out_of_circuit_plate": mod.OUT_OF_CIRCUIT_PLATE,
    }
