from unittest import TestCase

from kmer_processor import *


class Test(TestCase):
    def test_neighborhood_hamming_k2(self):
        self.assertEqual(hamming_neighborhood("AT", 0), {"AT"})
        self.assertEqual(hamming_neighborhood("AT", 1),
                         {'AA', 'AC', 'AG', 'AT', 'CT', 'GT', 'TT'})
        self.assertEqual(hamming_neighborhood("AT", 2),
                         {'CA', 'AT', 'GC', 'AC', 'GA', 'AG', 'CT', 'AA', 'TC',
                          'TG', 'TA', 'GT', 'GG', 'CC', 'TT', 'CG'})
        self.assertEqual(hamming_neighborhood("AT", 3),
                         {'CA', 'AT', 'GC', 'AC', 'GA', 'AG', 'CT', 'AA', 'TC',
                          'TG', 'TA', 'GT', 'GG', 'CC', 'TT', 'CG'})

    def test_neighborhood_hamming_k3(self):
        self.assertEqual(hamming_neighborhood("CAT", 0), {"CAT"})
        self.assertEqual(hamming_neighborhood("CAT", 1),
                         {'CCT', 'CAT', 'CTT', 'AAT', 'CAG', 'CAC', 'TAT',
                          'CGT', 'GAT', 'CAA'})
        self.assertEqual(hamming_neighborhood("CAT", 2),
                         {'GAC', 'CTA', 'GAG', 'CAA', 'GCT', 'TAC', 'CGA',
                          'AAT', 'CAG', 'CAC', 'CTG', 'CCC', 'GAT', 'CTC',
                          'TTT', 'TGT', 'CCG', 'CCT', 'CGC', 'GGT', 'CCA',
                          'CAT', 'CTT', 'AAC', 'TAA', 'AAA', 'GAA', 'ATT',
                          'TCT', 'TAG', 'ACT', 'TAT', 'CGT', 'CGG', 'GTT',
                          'AGT', 'AAG'})
