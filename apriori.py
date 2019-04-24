from itertools import chain
from itertools import combinations
import numpy as np
import scipy as sp


# Associative rule mining for dataset D with
# and s is support for k tuples
def apriori(D,s,k):
    # instantiate candidate set and count
    itemset,itemcount = np.unique( 
        D,return_counts=True
    )

    # helper function, returns the count of a 
    # set s as s appears in each row of itemset
    def contains(s):
        foundrows = [
            set(row).issuperset(s) 
            for row in itemset.tolist()
        ]
        count = len(np.where(foundrows)[0])
        return count

    # find cases that statisy support threshold
    np.where(itemcount < s, 0, itemcount)
    # items that meet support, frequent singles
    itemset[np.where(itemcount > 0)]   

    i = 2 # start i = 2 to build candidate set

    while i < k:
        # generate all possible combinations of 
        # remaining items in itemset
        canidates = np.fromiter(
            chain(combinations(itemset,i))
        )
        itemcount = np.zeros(shape=len(canidates))

        # assign the count of each subset to itemcount
        for r in len(canidates):
            itemcount[r] = contains(r)
        
        # TODO figure out indexing of subsets and itemsets


# takes a tuple T and returns a hashed index
def pcy_hash(T,n):
    # base case if only one element in in tuple
    if n == 1:
        return T[0]
    index = 0
    # TODO implement hash function
    return 0


# Associative rule mining for dataset D with
# and s is support for k tuples
def pcy(D,s,k):

    n = len(np.unique(D))
    buckets = np.zeros(
        shape=(n//2), dtype=int)  

    # helper function increments buckets
    # given an array of hash values
    @np.vectorize
    def increment_buckets(index):
        buckets[index] += 1

    # loop through finding frequent 
    # k-tuples of support s
    for i in range(k):
        for d in D:
            # generate canidate tuples
            combos = chain.from_iterable(combinations(d,i))
            combos = np.fromiter(combos,dtype=int)
            combos = combos.reshape((len(combos) // i),i)
            # hash each canidate tuple in combos and increment bucket
            hashed = np.where(combos,pcy_hash(combos,(n // i)),-1)
            increment_buckets(hashed)            


# TODO additional preprocessing dataset
# encode stocks to np.uint16 index
if __name__ == '__main__':
    pass