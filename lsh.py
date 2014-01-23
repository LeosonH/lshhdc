"""
lsh.py

Algorithms based on 'Mining of Massive Datasets'
"""
from unionfind import UnionFind
from collections import defaultdict
from collections import defaultdict, namedtuple
from copy import deepcopy
import operator


def shingle(s, k):
    """Generate k-length shingles of string s"""
    k = min(len(s), k)
    for i in range(len(s) - k + 1):
        yield s[i:i+k]

def hshingle(s, k):
    """Generate k-length shingles then hash"""
    for s in shingle(s, k):
        yield hash(s)

def jaccard_sim(X, Y):
    """Jaccard similarity between two sets"""
    x = set(X)
    y = set(Y)
    return float(len(x & y)) / len(x | y)

def jaccard_dist(X, Y):
    """Jaccard distance between two sets"""
    return 1 - jaccard_sim(X, Y)


class Signature(object):
    """Signature Base class."""

    def __init__(self, dim):
        self.dim = dim
        self.hashes = self.hash_functions()

    def hash_functions(self):
        """Returns dim different hash functions"""
        pass

    def sign(self, object):
        """Return the signature for object s"""
        pass


class MinHashSignature(Signature):
    """Creates signatures for sets/tuples using minhash."""

    def hash_functions(self):
        """Return dim different hash functions"""
        def hash_factory(n):
            return lambda x: hash("salt" + str(n) + str(x) + "salt")
        return [ hash_factory(_) for _ in range(self.dim) ]

    def sign(self, s):
        """Returns minhash signature for set s"""
        sig = [ float("inf") ] * self.dim
        for hash_ix, hash_fn in enumerate(self.hashes):
            sig[hash_ix] = min(hash_fn(value) for value in s)
        return sig


class LSH(object):
    """Locality sensitive hashing.  Uses a banding approach to hash
    similar signatures to the same buckets."""
    def __init__(self, length, threshold):
        self.length = length
        self.threshold = threshold
        self.bandwidth = self.get_bandwidth(length, threshold)

    def hash(self, sig, band_idx=None):
        """Generate hashvals for this signature"""
        for band in zip(*(iter(sig),) * self.bandwidth):
            yield hash("salt" + str(band) + "tlas")

    def get_bandwidth(self, n, t):
        """Approximates the bandwidth (number of rows in each band)
        needed to get threshold.

        Threshold t = (1/b) ** (1/r) where
        b = #bands
        r = #rows per band
        n = b * r = #elements in signature
        """

        best = n, 1
        minerr  = float("inf")
        for r in range(1, n + 1):
            try:
                b = 1. / (t ** r)
            except:             # Divide by zero, your signature is huge
                return best
            err = abs(n - b * r)
            if err < minerr:
                best = r
                minerr = err
        return best

    def get_threshold(self):
        r = self.bandwidth
        b = self.length / r
        return (1. / b) ** (1. / r)

    def get_n_bands(self):
        return int(self.length / self.bandwidth)


class Cluster(object):
    """Clusters sets with Jaccard similarity above threshold with high
    probability.

    Algorithm based on Rajaraman, "Mining of Massive Datasets":
    1. Generate set signature
    2. Use LSH to map similar signatures to same buckets
    3. Use UnionFind to merge buckets containing same values
    """
    def __init__(self, width=10, threshold=0.5):
        self.width = width
        self.unionfind = UnionFind()
        self.signer = MinHashSignature(width)
        self.hasher = LSH(width, threshold)
        self.hashmaps = [defaultdict(list)
                         for _ in range(self.hasher.get_n_bands())]

    def add_set(self, s, label=None):
        # A label for this set
        if not label:
            label = s

        # Add to unionfind structure
        self.unionfind[label]

        # Get signature
        sig = self.signer.sign(s)

        # Union labels with same LSH key in same band
        for band_idx, hshval in enumerate(self.hasher.hash(sig)):
            self.hashmaps[band_idx][hshval].append(label)
            self.unionfind.union(label, self.hashmaps[band_idx][hshval][0])

    def get_sets(self):
        return self.unionfind.sets()


class ConstrainedCluster(Cluster):

    """To fight the problem of big clusters created by the aggregation of a
    large number of false positives (i.e. two items found to be a candidate
    pair, but that really shouldn't belong to the same cluster), this class
    introduces an extra constraint which must be met for two items to be
    clustered. This mechanism imposes that we keep track of extra items, that
    are encapsulated in the LabelObj namedtuple. The constraint, by default, is
    that the Jaccard Similarity must be as high as the hasher threshold, which
    is defined with this anonymous function:

    lambda lo1, lo2: jaccard_sim(lo1.obj, lo2.obj)

    where the lo's are object of type LabelObj. However, this could be easily
    redefined to a function possibly more useful in some context, like the
    Levenshtein Ratio for instance (or any other similarity function to be
    maximized):

    lambda lo1, lo2: Levenshtein.ratio(lo1.obj, lo2.obj)

    which will work, provided that an "obj" argument has been previously passed
    to add_set.  In this case "obj" is a string, but it could be of whatever
    type, as long as the "contraint_fn" function properly handles it.
    """

    # Structure to be stored in the ConstrainedCluster.hashmaps band/hash cell
    # cluster lists.
    LabelObj = namedtuple('LabelObj', 'label obj')

    def __init__(self, width=10, threshold=0.5,
                 constraint_min=None,
                 constraint_fn=lambda lo1, lo2:
                                   jaccard_sim(lo1.obj, lo2.obj)):
        super(ConstrainedCluster, self).__init__(width, threshold)
        if constraint_min is None:
            self.constraint_min = threshold
        else:
            self.constraint_min = constraint_min
        self.constraint_fn = constraint_fn
        # Note that self.hashmaps, although having the same structure as in the
        # parent class, is used quite differently here: each band/hash cell now
        # corresponds to a list of lists (instead of a single list). Each list
        # contains at least one LabelSetObj instance, and will possibly grow
        # when hash collisions occur. However, to be fused within a certain
        # list, an item must be similar enough to its first item (i.e. the
        # constraint must be satisfied). If no list is found with an item to
        # satisfy the constraint, a new list with the element is simply appended
        # to the band/hash cell.

    def add_set(self, s, label=None, obj=None):
        # A label for this set
        if not label:
            label = s

        # if obj is not defined, s is used
        lo = ConstrainedCluster.LabelObj(label, obj if obj else s)

        # Add to unionfind structure
        self.unionfind[label]

        # Get signature
        sig = self.signer.sign(s)

        # Union labels with same LSH key in same band that satisfy constraint
        for band_idx, hshval in enumerate(self.hasher.hash(sig)):
            # apply the constraint function to compare the current element
            # to every first element of every candidate clusters
            jsc = [(self.constraint_fn(lo, cluster[0]), cluster)
                   for cluster in self.hashmaps[band_idx][hshval]]
            # retain the best (if it exists) of those over the threshold
            jsc = sorted([(js, cluster) for js, cluster in jsc
                          if js >= self.constraint_min], reverse=True)
            if jsc:
                cluster = jsc[0][1]
                cluster.append(deepcopy(lo))
                # the candidate pair is now clustered
                self.unionfind.union(lo.label, cluster[0].label)
            else:
                # no clustering is performed
                self.hashmaps[band_idx][hshval].append([deepcopy(lo)])


class SemiParallellizableConstrainedCluster(Cluster):

    """This is a semi-parallel version of ConstrainedCluster, to be used with
    multiprocessing; explanations and documentation soon to come..
    """

    def __init__(self, width=10, threshold=0.5,
                 constraint_min=None,
                 constraint_fn=lambda lo1, lo2:
                                   jaccard_sim(lo1.obj, lo2.obj),
                 sigmaps_to_merge=None):
        super(SemiParallellizableConstrainedCluster, self).__init__(width, threshold)
        if constraint_min is None:
            self.constraint_min = threshold
        else:
            self.constraint_min = constraint_min
        self.constraint_fn = constraint_fn
        # Note that self.hashmaps, although having the same structure as in the
        # parent class, is used quite differently here: each band/hash cell now
        # corresponds to a list of lists (instead of a single list). Each list
        # contains at least one LabelSetObj instance, and will possibly grow
        # when hash collisions occur. However, to be fused within a certain
        # list, an item must be similar enough to its first item (i.e. the
        # constraint must be satisfied). If no list is found with an item to
        # satisfy the constraint, a new list with the element is simply appended
        # to the band/hash cell.
        if sigmaps_to_merge is None:
            self.sigmap = {}
        else:
            self.sigmap = dict(reduce(operator.__add__,
                                      [sm.items() for sm in sigmaps_to_merge]))

    def sign(self, s, label=None, obj=None):
        # A label for this set
        if not label:
            label = s
        self.sigmap[label] = (self.signer.sign(s) if s else None,
                              obj if obj else s)

    def find_clusters(self):
        for label, (sig, obj) in self.sigmap.iteritems():
            self.unionfind[label]
            if sig is None: continue
            lo = ConstrainedCluster.LabelObj(label, obj)

            # Union labels with same LSH key in same band that satisfy constraint
            for band_idx, hshval in enumerate(self.hasher.hash(sig)):
                # apply the constraint function to compare the current element
                # to every first element of every candidate clusters
                jsc = [(self.constraint_fn(lo, cluster[0]), cluster)
                       for cluster in self.hashmaps[band_idx][hshval]]
                # retain the best (if it exists) of those over the threshold
                jsc = sorted([(js, cluster) for js, cluster in jsc
                              if js >= self.constraint_min], reverse=True)
                if jsc:
                    cluster = jsc[0][1]
                    cluster.append(deepcopy(lo))
                    # the candidate pair is now clustered
                    self.unionfind.union(lo.label, cluster[0].label)
                else:
                    # no clustering is performed
                    self.hashmaps[band_idx][hshval].append([deepcopy(lo)])


if __name__ == '__main__':

    n = 2
    sa = set(shingle("1234abcdef", n))
    sb = set(shingle("4321abcdef", n))

    print 'Jaccard Sim:', jaccard_sim(sa, sb)

    cluster = Cluster()
    cluster.add_set(sa, 'a')
    cluster.add_set(sb, 'b')
    print 'Cluster:', cluster.get_sets() # [['a', 'b']]

    cluster = ConstrainedCluster()
    cluster.add_set(sa, 'a')
    cluster.add_set(sb, 'b')
    print 'ConstrainedCluster:', cluster.get_sets() # [['a'], ['b']]
