"""The age-dependent point cloud protocol.

This is the seam between gtrack and its consumers. gtrack answers "what
labelled cloud exists at age X"; a consumer answers "what do I do with it".
Anything that can answer the first question satisfies :class:`AgeCloudSource`,
including a plain numpy test double, so a consumer never has to import a gtrack
class or construct plate-reconstruction machinery to be tested.

The protocol is deliberately small. Everything a caller needs to drive a source
across geological time is here, and nothing about how the cloud is produced is.
"""

from typing import Protocol, runtime_checkable

from .point_rotation import PointCloud


@runtime_checkable
class AgeCloudSource(Protocol):
    """A source of labelled point clouds indexed by geological age.

    Attributes
    ----------
    provides : frozenset of str
        Names of the properties every returned cloud carries, EXCLUDING the
        coordinates themselves. A consumer reads this to know which
        ``get_property`` calls will succeed. Coordinates are always present
        as ``PointCloud.xyz`` and are never listed here.
    monotonic_backward : bool, default=False
        True when the source may only be walked backwards in time — each
        ``at_age`` call must ask for an age at least as old as the last one,
        because the underlying state is advanced in place and cannot be
        rewound. A forward-only ocean tracker sets this; a source that
        recomputes from scratch at every age leaves it False. A consumer reads
        this single boolean rather than re-implementing the walk contract.

    Notes
    -----
    ``runtime_checkable`` makes ``isinstance`` work, and it checks for the
    presence of the members rather than their types or signatures. That is
    enough to catch the mistake worth catching — passing something that is not
    a source at all — and it keeps a numpy double viable as a stand-in.

    Note that the attributes must exist on the object, not merely be annotated
    on its class: an annotation without a value creates no attribute, and
    ``isinstance`` will return False. Set them as class constants or in
    ``__init__``.

    Examples
    --------
    >>> class ConstantCloud:
    ...     provides = frozenset({"thickness"})
    ...     monotonic_backward = False
    ...     def at_age(self, age):
    ...         return _some_cloud
    ...     def validate_age(self, age):
    ...         pass
    >>> isinstance(ConstantCloud(), AgeCloudSource)
    True
    """

    provides: frozenset
    monotonic_backward: bool = False

    def at_age(self, age: float) -> PointCloud:
        """Return the labelled cloud at ``age``.

        Parameters
        ----------
        age : float
            Geological age in Ma, non-negative, larger meaning older.

        Returns
        -------
        PointCloud
            A cloud carrying every property named in ``provides``.
        """
        ...

    def validate_age(self, age: float) -> None:
        """Raise if ``age`` is outside what this source can answer for.

        The default is to accept everything. Implementations that own a plate
        model or a forward-only walk override it to reject ages beyond the
        model's range, or younger than the walk has already reached.

        Callers are expected to invoke this before any collective or
        expensive work, so that a rejection is raised everywhere rather than
        on one process.

        Parameters
        ----------
        age : float
            Geological age in Ma.
        """
        ...
