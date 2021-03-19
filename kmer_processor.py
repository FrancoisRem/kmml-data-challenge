DNA_NUCLEOTIDES = {'A', 'T', 'G', 'C'}


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
