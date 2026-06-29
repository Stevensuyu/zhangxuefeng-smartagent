"""Collector contracts for reviewed raw data snapshots."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.data_pipeline.raw_store.manifest import RawSnapshotManifest


@dataclass(frozen=True)
class CollectedSnapshot:
    """Validated local snapshot metadata returned by a collector."""

    root_dir: Path
    manifest: RawSnapshotManifest
    file_issues: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether all manifest files exist and match checksums."""
        return not self.file_issues


class SnapshotCollector(Protocol):
    """Collector interface that returns a local raw snapshot.

    Collectors can fetch remote content and execute crawler or API behavior
    when approved through the review process.
    """

    def collect(self) -> CollectedSnapshot:
        """Return one local raw snapshot and its manifest validation result."""
