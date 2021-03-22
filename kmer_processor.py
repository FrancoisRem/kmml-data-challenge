from functools import partial

import numpy as np
from tqdm import tqdm

DNA_NUCLEOTIDES = {'A', 'T', 'G', 'C'}
tqdm.pandas()


class KMerProcessor:
    def __init__(self, seq_series):
        """
        :param df: pandas series containing DNA sequences in the alphabet
         {'A', 'B', 'C', 'D'} and with fixed length L.
        """
        self.seq_series = seq_series
        self.kmers_support = set()
        self.kmers_neighborhood_cache = {}

    def _compute_kmer_mismatch_for_seq(self, seq, k, m):
        # spectrum = defaultdict(int)
        spectrum = {}
        seq_length = len(seq)
        for i in range(seq_length - k + 1):
            kmer = seq[i:i + k]
            if kmer not in self.kmers_neighborhood_cache:
                self.kmers_neighborhood_cache[kmer] = hamming_neighborhood(
                    seq[i:i + k], m) & self.kmers_support

            for neighbor in self.kmers_neighborhood_cache[kmer]:
                if neighbor in spectrum:
                    spectrum[neighbor] += 1
                else:
                    spectrum[neighbor] = 1
        return spectrum

    def compute_kmer_mismatch(self, k, m):
        self.kmers_support = all_kmers_in_sequences(self.seq_series, k)

        worker = partial(self._compute_kmer_mismatch_for_seq, k=k, m=m)
        print("Computing kmer mismatch spectrum:")
        spectrum_series = self.seq_series.progress_apply(worker)

        return spectrum_series


def all_kmers_in_sequences(seq_series, k):
    """
    Compute the set of all the kmers contained in all the sequences from
    seq_series.
    :param seq_series: pandas series containing DNA sequences in the alphabet
    DNA_NUCLEOTIDES and with fixed length L.
    :param k: int, length of the kmers to consider, must be >= 0
    :return: set
    """
    all_kmers = set()
    seq_length = len(seq_series[0])

    def append_to_kmer_set(seq):
        for i in range(seq_length - k + 1):
            all_kmers.add(seq[i:i + k])

    seq_series.apply(append_to_kmer_set)
    return all_kmers


def hamming_neighborhood(kmer, max_distance):
    """
    Compute the set of kmers within max_distance of kmer according to the
    Hamming distance over the alphabet DNA_NUCLEOTIDES.
    Implementation derived from https://stackoverflow.com/a/50893025.
    :param kmer: string of characters in DNA_NUCLEOTIDES
    :param max_distance: int, the maximum distance to be in the neighborhood,
    must be >= 0.
    :return: set
    """
    kmer_length = len(kmer)
    neighborhood = set()
    allbut = {nucl: [other_nucl for other_nucl in DNA_NUCLEOTIDES if
                     other_nucl != nucl] for nucl in DNA_NUCLEOTIDES}

    def inner(i, remaining_distance):
        if not remaining_distance or i == kmer_length:
            neighborhood.add("".join(pattern))
            return
        inner(i + 1, remaining_distance)
        current_nucleotide = pattern[i]
        for pattern[i] in allbut[current_nucleotide]:
            inner(i + 1, remaining_distance - 1)
        pattern[i] = current_nucleotide

    pattern = list(kmer)
    inner(0, max_distance)
    return neighborhood


def index_kmers(kmers_support):
    """
    Index all kmers in kmers_support by assigning to each of them a unique
    integer in [0, len(kmers_support))
    :param kmers_support: set, contains kmers overs the alphabet DNA_NUCLEOTIDES
    with fixed length k.
    :return: dict, map from kmer to index
    """
    kmer_to_index = {}
    for i, x in enumerate(kmers_support):
        kmer_to_index[x] = i
    return kmer_to_index


def compute_spectrums_matrix(kmers_support, spectrums):
    """
    Compute the matrix of the spectrums (dict kmer -> frequencies) as a dense
    numpy array.
    Note: the return array is dense and can take a lot of space in RAM, use with
    special care.
    :param kmers_support: set, set of all the kmers contained in all the
    elements of spectrums.
    :param spectrums:
    :return: (n, d)-np.array with n = len(spectrums) and d = len(kmer_support)
    """
    kmer_to_index = index_kmers(kmers_support)
    nb_sequences = len(spectrums)
    nb_kmers = len(kmer_to_index)
    spectrums_matrix = np.zeros((nb_sequences, nb_kmers))
    print("Computing dense spectrums matrix:")
    for i, spectrum in enumerate(tqdm(spectrums)):
        for kmer, freq in spectrum.items():
            spectrums_matrix[i, kmer_to_index[kmer]] = freq
    return spectrums_matrix
