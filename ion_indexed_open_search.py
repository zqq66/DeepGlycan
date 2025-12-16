"""
Ion-indexed open modification search implemented in Python.

Key references to MetaMorpheus source:
    - Fragment index construction mirrors the logic in
      MetaMorpheus/MetaMorpheus/EngineLayer/Indexing/IndexingEngine.cs::RunSpecific.
    - First-pass candidate collection follows
      MetaMorpheus/MetaMorpheus/EngineLayer/ModernSearch/ModernSearchEngine.cs::IndexScoreScan
      and ::IncrementPeptideScoresInBin.
    - Fine scoring and match reporting are simplified from
      MetaMorpheus/MetaMorpheus/EngineLayer/ModernSearch/ModernSearchEngine.cs::FineScorePeptide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import bisect
import math
import pathlib


# Monoisotopic masses for the 20 canonical residues (in Daltons).
AMINO_ACID_MASSES: Dict[str, float] = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

PROTON_MASS = 1.007276466812
WATER_MASS = 18.010564684
AMMONIA_MASS = 17.0265491015


@dataclass
class PeptideEntry:
    """Container for peptide meta-data."""

    sequence: str
    mass: float
    fragments: Tuple[float, ...]
    fragment_cleavages: Tuple[int, ...]


@dataclass(order=True)
class PeptideMatch:
    """Represents one peptide-spectrum match."""
    normalized_matched_ions: float
    normalized_score: float
    score: float
    matched_ions: int
    intensity_sum: float
    peptide_index: int
    sequence: str
    peptide_mass: float


class OpenSearchMassDiffAcceptor:
    """
    Simplified open search mass-difference acceptor.

    Based on MetaMorpheus' open search acceptor in
    MetaMorpheus/MetaMorpheus/EngineLayer/PrecursorSearchModes/OpenMassDiffAcceptor.cs::OpenSearchMode.
    The MetaMorpheus implementation is fully unbounded; this Python facade allows setting
    an optional symmetric absolute mass window for practical runtime limits.
    """

    def __init__(self, max_abs_mass_difference: float | None = None) -> None:
        self.max_abs_mass_difference = max_abs_mass_difference

    def accepts(self, precursor_mass: float, peptide_mass: float) -> bool:
        if self.max_abs_mass_difference is None:
            return True
        return abs(precursor_mass - peptide_mass) <= self.max_abs_mass_difference

    def allowed_interval_from_observed_mass(self, _: float) -> Tuple[float, float]:
        if self.max_abs_mass_difference is None:
            return (-math.inf, math.inf)
        return (-self.max_abs_mass_difference, self.max_abs_mass_difference)


class IonIndexedOpenSearch:
    """
    Fragment-ion-indexed peptide-first open modification search.

    The in-silico digestion and fragment index construction mimic the flow in
    EngineLayer/Indexing/IndexingEngine.cs::RunSpecific, while the scan scoring logic draws from
    EngineLayer/ModernSearch/ModernSearchEngine.cs::IndexScoreScan.
    """

    FRAGMENT_BINS_PER_DA = 1000

    def __init__(
        self,
        peptides: Sequence[PeptideEntry],
        mass_acceptor: OpenSearchMassDiffAcceptor | None = None,
        score_threshold: int = 0,
        include_cz_ions: bool = True,
    ) -> None:
        if not peptides:
            raise ValueError("Cannot initialize search without peptides.")

        # Ensure peptides are sorted by monoisotopic mass, mirroring the C# indexer.
        self.peptides: List[PeptideEntry] = sorted(peptides, key=lambda p: p.mass)
        self.mass_acceptor = mass_acceptor or OpenSearchMassDiffAcceptor()
        self.score_threshold = score_threshold
        self.include_cz_ions = include_cz_ions

        self.fragment_index = self._build_fragment_index(self.peptides)
        self.fragment_bins = sorted(self.fragment_index.keys())
        self.max_fragment_mass = max(
            mass for peptide in self.peptides for mass in peptide.fragments
        )

    @classmethod
    def from_fasta(
        cls,
        fasta_path: str | pathlib.Path,
        enzyme: str = "trypsin",
        max_missed_cleavages: int = 2,
        min_length: int = 6,
        max_length: int = 40,
        mass_acceptor: OpenSearchMassDiffAcceptor | None = None,
        score_threshold: int = 3,
        include_cz_ions: bool = True,
    ) -> "IonIndexedOpenSearch":
        """
        Factory that digests a FASTA database to generate peptide entries and build the fragment index.

        The digestion procedure is a Python analogue of the MetaMorpheus in-silico digestion loop in
        EngineLayer/Indexing/IndexingEngine.cs::RunSpecific.
        """

        fasta_path = pathlib.Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")

        proteins = list(_read_fasta_sequences(fasta_path))
        peptides: List[PeptideEntry] = []
        for protein_seq in proteins:
            peptides.extend(
                _digest_sequence(
                    protein_seq,
                    enzyme=enzyme,
                    max_missed_cleavages=max_missed_cleavages,
                    min_length=min_length,
                    max_length=max_length,
                    include_cz_ions=include_cz_ions,
                )
            )

        return cls(
            peptides,
            mass_acceptor=mass_acceptor,
            score_threshold=score_threshold,
            include_cz_ions=include_cz_ions,
        )

    # --- Index construction helpers --------------------------------------------------------------

    def _build_fragment_index(
        self, peptides: Sequence[PeptideEntry]
    ) -> Dict[int, List[int]]:
        """
        Build the fragment-ion index keyed by coarse-grained mass bins.

        Directly mirrors the fragment population code in
        EngineLayer/Indexing/IndexingEngine.cs::RunSpecific by looping over peptides (sorted
        by mass) and assigning them to fragment bins.
        """

        fragment_index: Dict[int, List[int]] = {}

        for peptide_id, peptide in enumerate(peptides):
            for fragment_mass in peptide.fragments:
                if fragment_mass <= 0:
                    continue

                fragment_bin = int(round(fragment_mass * self.FRAGMENT_BINS_PER_DA))
                fragment_index.setdefault(fragment_bin, []).append(peptide_id)
        return fragment_index

    # --- Public API ------------------------------------------------------------------------------

    def find_top_k_peptides(
        self,
        spectrum: Sequence[Tuple[float, float]],
        precursor_mass: float | None,
        top_k: int = 5,
        fragment_tolerance: float = 0.02,
    ) -> List[PeptideMatch]:
        """
        Score the supplied spectrum against the indexed database and return the top-k matches.

        The first-pass candidate collection is a Python port of
        ModernSearchEngine.IndexScoreScan/IncrementPeptideScoresInBin. Candidates that
        reach the coarse score threshold proceed to a fine scoring pass (simplified from
        ModernSearchEngine.FineScorePeptide).

        Parameters
        ----------
        spectrum:
            Sequence of (m/z, intensity) centroided peaks (e.g., a Python list of tuples or a
            2-D NumPy array converted via `.tolist()`). The current implementation scores against
            singly protonated b/c/y/z fragment ions, so the supplied m/z values should be comparable
            (e.g., either charge-reduced to +1 or already measured at +1).
        precursor_mass:
            Observed precursor monoisotopic mass (float). Pass `None` to skip precursor filtering
            and perform a fully open fragment-ion search.

        Returns
        -------
        List[PeptideMatch]
            Matches ranked by normalized_score (raw score divided by peptide length). Raw scores
            and matched-ion counts are retained for downstream inspection.
        """

        if not spectrum:
            return []

        scoring_table = [0] * len(self.peptides)
        candidate_ids: set[int] = set()

        if precursor_mass is None:
            lowest_peptide_mass = -math.inf
            highest_peptide_mass = math.inf
        else:
            allowed_min, allowed_max = (
                self.mass_acceptor.allowed_interval_from_observed_mass(precursor_mass)
            )
            lowest_peptide_mass = (
                precursor_mass + allowed_min
                if not math.isinf(allowed_min)
                else -math.inf
            )
            highest_peptide_mass = (
                precursor_mass + allowed_max
                if not math.isinf(allowed_max)
                else math.inf
            )

        for mz, _ in spectrum:
            if mz <= 0:
                continue

            min_bin = int(
                max(0, math.floor((mz - fragment_tolerance) * self.FRAGMENT_BINS_PER_DA))
            )
            max_bin = int(
                math.ceil((mz + fragment_tolerance) * self.FRAGMENT_BINS_PER_DA)
            )

            left = bisect.bisect_left(self.fragment_bins, min_bin)
            right = bisect.bisect_right(self.fragment_bins, max_bin)

            for fragment_bin in self.fragment_bins[left:right]:
                bin_hits = self.fragment_index[fragment_bin]

                start, end = self._get_first_and_last_indexes(
                    lowest_peptide_mass, highest_peptide_mass, bin_hits
                )

                if start > end:
                    continue

                for idx in bin_hits[start : end + 1]:
                    scoring_table[idx] += 1
                    if scoring_table[idx] == self.score_threshold:
                        candidate_ids.add(idx)

        if not candidate_ids:
            # print('scoring_table',scoring_table)
            # Fall back to any peptide that collected at least one indexed match.
            candidate_ids = {idx for idx, score in enumerate(scoring_table) if score > 3}

        if not candidate_ids:
            return []
        # print('len(candidate_ids)',len(candidate_ids))
        # Fine scoring of surviving candidates.
        matches: List[PeptideMatch] = []
        for idx in candidate_ids:
            peptide = self.peptides[idx]
            if precursor_mass is not None and not self.mass_acceptor.accepts(
                precursor_mass, peptide.mass
            ):
                continue

            matched_ions, intensity_sum, residue_matches = _fine_score(
                peptide.fragments,
                peptide.fragment_cleavages,
                spectrum,
                fragment_tolerance,
            )
            if matched_ions == 0:
                continue

            score = intensity_sum if intensity_sum > 0 else float(matched_ions)
            length = max(len(peptide.sequence), 1)
            normalized_matched = residue_matches / length
            normalized_score = score / length
            matches.append(
                PeptideMatch(
                    normalized_score=normalized_score,
                    normalized_matched_ions=normalized_matched,
                    score=score,
                    matched_ions=matched_ions,
                    intensity_sum=intensity_sum,
                    peptide_index=idx,
                    sequence=peptide.sequence,
                    peptide_mass=peptide.mass,
                )
            )

        # matches = [match for match in matches if match.normalized_matched_ions >= 0.5]
        matches.sort(reverse=True)
        return matches[:top_k]

    # --- Bin search (BinarySearchBinForPrecursorIndex analogue) ----------------------------------

    def _get_first_and_last_indexes(
        self,
        lowest_peptide_mass_to_look_for: float,
        highest_peptide_mass_to_look_for: float,
        bin_hits: Sequence[int],
    ) -> Tuple[int, int]:
        """
        Port of ModernSearchEngine.GetFirstAndLastIndexesInBinToIncrement.
        """

        start = 0
        end = len(bin_hits) - 1

        if not math.isinf(highest_peptide_mass_to_look_for):
            end = self._binary_search_bin(bin_hits, highest_peptide_mass_to_look_for)
            end = min(end, len(bin_hits) - 1)

        if not math.isinf(lowest_peptide_mass_to_look_for):
            start = self._binary_search_bin(bin_hits, lowest_peptide_mass_to_look_for)
            start = min(start, len(bin_hits) - 1)

        while start <= end and self.peptides[bin_hits[start]].mass < lowest_peptide_mass_to_look_for:
            start += 1

        while end >= start and self.peptides[bin_hits[end]].mass > highest_peptide_mass_to_look_for:
            end -= 1

        return start, end

    def _binary_search_bin(
        self, bin_hits: Sequence[int], peptide_mass_to_look_for: float
    ) -> int:
        """
        Mimics ModernSearchEngine.BinarySearchBinForPrecursorIndex.
        """

        left = 0
        right = len(bin_hits) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_mass = self.peptides[bin_hits[mid]].mass

            if right - left < 2:
                for idx in range(right, -1, -1):
                    if self.peptides[bin_hits[idx]].mass <= peptide_mass_to_look_for:
                        return idx
                break

            if peptide_mass_to_look_for > mid_mass:
                left = mid + 1
            else:
                right = mid - 1

        return 0


# --------------------------------------------------------------------------------------------------
# FASTA parsing and digestion helpers
# --------------------------------------------------------------------------------------------------


def _read_fasta_sequences(fasta_path: pathlib.Path) -> Iterable[str]:
    current: List[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    yield "".join(current)
                    current.clear()
            else:
                current.append(line)
        if current:
            yield "".join(current)


def _digest_sequence(
    sequence: str,
    enzyme: str,
    max_missed_cleavages: int,
    min_length: int,
    max_length: int,
    include_cz_ions: bool,
) -> List[PeptideEntry]:
    if enzyme.lower() != "trypsin":
        raise NotImplementedError("Only trypsin digestion is currently implemented.")

    cleavage_points = _trypsin_cleavage_sites(sequence)
    peptides: List[PeptideEntry] = []

    for start_idx in range(len(cleavage_points) - 1):
        for missed in range(max_missed_cleavages + 1):
            end_idx = start_idx + missed + 1
            if end_idx >= len(cleavage_points):
                break

            start = cleavage_points[start_idx]
            end = cleavage_points[end_idx]
            peptide_seq = sequence[start:end]

            if not (min_length <= len(peptide_seq) <= max_length):
                continue

            if any(residue not in AMINO_ACID_MASSES for residue in peptide_seq):
                continue

            mass = _monoisotopic_mass(peptide_seq)
            fragments, cleavages = _theoretical_fragments(peptide_seq, include_cz_ions)
            peptides.append(PeptideEntry(peptide_seq, mass, fragments, cleavages))

    return peptides


def _trypsin_cleavage_sites(sequence: str) -> List[int]:
    sites = [0]
    for idx, residue in enumerate(sequence):
        if residue in ("K", "R"):
            next_residue = sequence[idx + 1] if idx + 1 < len(sequence) else None
            if next_residue != "P":
                sites.append(idx + 1)
    if sites[-1] != len(sequence):
        sites.append(len(sequence))
    return sites


def _monoisotopic_mass(peptide: str) -> float:
    return sum(AMINO_ACID_MASSES[residue] for residue in peptide) + WATER_MASS


def _theoretical_fragments(
    peptide: str, include_cz_ions: bool = True
) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    fragment_masses: List[float] = []
    cleavage_indexes: List[int] = []

    if len(peptide) < 2:
        return tuple(), tuple()

    running_mass = PROTON_MASS
    for idx, residue in enumerate(peptide[:-1], start=1):
        running_mass += AMINO_ACID_MASSES[residue]
        fragment_masses.append(running_mass)
        cleavage_indexes.append(idx)
        if include_cz_ions:
            fragment_masses.append(running_mass + AMMONIA_MASS)
            cleavage_indexes.append(idx)

    running_mass = PROTON_MASS + WATER_MASS
    for suffix_idx, residue in enumerate(peptide[::-1][:-1], start=1):
        running_mass += AMINO_ACID_MASSES[residue]
        cleavage_index = len(peptide) - suffix_idx
        fragment_masses.append(running_mass)
        cleavage_indexes.append(cleavage_index)
        if include_cz_ions:
            z_mass = running_mass - AMMONIA_MASS
            if z_mass > 0:
                fragment_masses.append(z_mass)
                cleavage_indexes.append(cleavage_index)

    return tuple(fragment_masses), tuple(cleavage_indexes)


# --------------------------------------------------------------------------------------------------
# Fine scoring helpers
# --------------------------------------------------------------------------------------------------


def _fine_score(
    fragments: Sequence[float],
    fragment_cleavages: Sequence[int],
    spectrum: Sequence[Tuple[float, float]],
    fragment_tolerance: float,
) -> Tuple[int, float, int]:
    matched_ions = 0
    intensity_sum = 0.0
    matched_cleavages: set[int] = set()

    for fragment_mass, cleavage_index in zip(fragments, fragment_cleavages):
        for mz, intensity in spectrum:
            if abs(mz - fragment_mass) <= fragment_tolerance:
                matched_ions += 1
                intensity_sum += intensity
                if cleavage_index > 0:
                    matched_cleavages.add(cleavage_index)
                break

    return matched_ions, intensity_sum, len(matched_cleavages)


__all__ = [
    "IonIndexedOpenSearch",
    "OpenSearchMassDiffAcceptor",
    "PeptideEntry",
    "PeptideMatch",
]
