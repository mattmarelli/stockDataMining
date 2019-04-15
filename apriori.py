from itertools import chain
from itertools import combinations
import numpy as np
import scipy as sp


# Associative rule mining for dataset D with
# and s is support for k tuples
def apriori(D,s,k):
    # instantiate candidate set and count
    itemset,itemcount = np.unique( D,
        return_counts=True
    )
    # find cases that statisy support threshold
    np.where(itemcount < s, 0, itemcount)
    # items that meet support, frequent singles
    itemset[np.where(itemcount > 0)]   

    i = 2 # start i = 2 to build candidate set

    while i < k:
        canidates = np.fromiter(
            chain(combinations(itemset,i))
        )
        # TODO count the occurances in D of each
        # subset in candidate set and assign to itemset

     


# TODO additional preprocessing dataset
# encode stocks to np.uint16 index
if __name__ == '__main__':
    pass