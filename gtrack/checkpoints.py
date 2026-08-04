"""Checkpoint naming and selection policy.

gtrack writes the checkpoints, so gtrack owns the convention for naming them
and for choosing which one to resume from. Keeping that in one frozen object
means a consumer never has to reproduce the filename format or the
smallest-age-not-younger search, and the two cannot drift apart.

The policy is pure: it decides paths and answers questions about a directory.
It never writes a checkpoint itself — that stays with whatever owns the state
being checkpointed.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CheckpointPolicy:
    """Where checkpoints live, what they are called, and how often to write.

    Parameters
    ----------
    directory : Path
        Directory holding the checkpoint files. Not created here; whoever
        writes checkpoints is expected to have created it up front, so that a
        mistyped path fails at wiring time rather than mid-walk.
    interval_myr : float
        Minimum age separation between saves, in Myr. See :meth:`is_due`.
    prefix : str, default="ocean_checkpoint"
        Filename stem prefix. Files are ``<prefix>_<age>Ma.npz`` with an
        integer age.

    Examples
    --------
    >>> policy = CheckpointPolicy(Path("checkpoints"), interval_myr=50.0)
    >>> policy.path_for(123.4).name
    'ocean_checkpoint_123Ma.npz'
    """

    directory: Path
    interval_myr: float
    prefix: str = "ocean_checkpoint"

    def __post_init__(self):
        if self.interval_myr <= 0:
            raise ValueError(
                f"interval_myr must be positive, got {self.interval_myr}"
            )
        if not self.prefix:
            raise ValueError("prefix must not be empty")

    @property
    def _pattern(self) -> "re.Pattern":
        """Regex matching a checkpoint filename stem, capturing its age.

        Matched against ``Path.stem``, so the ``.npz`` suffix is not part of
        it; the suffix is checked separately.
        """
        return re.compile(rf"^{re.escape(self.prefix)}_(\d+)Ma$")

    def path_for(self, age: float) -> Path:
        """Return the path a checkpoint at ``age`` is written to.

        The age is rounded to a whole Myr, which is what makes the filenames
        parseable on the way back in.

        Parameters
        ----------
        age : float
            Geological age in Ma.

        Returns
        -------
        Path
            ``directory / "<prefix>_<age>Ma.npz"``.
        """
        return self.directory / f"{self.prefix}_{int(round(age))}Ma.npz"

    def best_at_or_before(self, target_age: float) -> Optional[Path]:
        """Return the best checkpoint to resume a walk towards ``target_age``.

        "At or before" is meant in TIME, not in the value of the age: the best
        checkpoint is the one with the smallest age that is still at least
        ``target_age``, i.e. the youngest checkpoint no younger than the
        target. Since a walk runs backwards through time and can only be
        advanced, resuming from the youngest such checkpoint minimises the
        stepping still to do. A larger age number is earlier in time.

        Parameters
        ----------
        target_age : float
            Geological age in Ma the caller wants to reach.

        Returns
        -------
        Path or None
            The chosen checkpoint, or None when the directory holds no
            candidate at or before ``target_age``.

        Notes
        -----
        A missing directory raises rather than returning None. Whoever writes
        checkpoints creates the directory at construction time, so a directory
        that has since disappeared is a real fault and should not be silently
        read as "no checkpoints yet".
        """
        pattern = self._pattern
        best_path = None
        best_age = None
        for candidate in self.directory.iterdir():
            if candidate.suffix != ".npz":
                continue
            match = pattern.match(candidate.stem)
            if match is None:
                continue
            age = int(match.group(1))
            if age < target_age:
                continue
            if best_age is None or age < best_age:
                best_age = age
                best_path = candidate
        return best_path

    def is_due(self, age: float, last_saved_age: Optional[float]) -> bool:
        """Report whether a checkpoint should be written at ``age``.

        Parameters
        ----------
        age : float
            Geological age in Ma currently reached.
        last_saved_age : float or None
            Age of the most recent save, or None if nothing has been saved
            yet in this run.

        Returns
        -------
        bool
            True when nothing has been saved yet, or when ``age`` differs
            from the last save by at least ``interval_myr``. The comparison
            is on absolute separation, so the cadence holds whichever
            direction the walk is running.
        """
        if last_saved_age is None:
            return True
        return abs(last_saved_age - age) >= self.interval_myr
