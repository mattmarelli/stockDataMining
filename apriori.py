from itertools import chain
from itertools import combinations
import numpy as np
import scipy.sparse as sp


# takes a tuple T and returns a hashed index
def pcy_hash(n,T):

    @np.vectorize
    def hash_items(item):
        return hash(item)

    # TODO perhaps write a better hash
    indices = hash_items(T)
    index = np.sum(indices)
    index = index % n
    return index


#returns all possible combinations 
# within array A of size n
def itemcombos(A,n):
    combos = chain.from_iterable(combinations(A,n))
    combos = np.fromiter(combos,dtype=int)
    combos = combos.reshape((len(combos) // n),n)
    return combos


# Associative rule mining for dataset D as a 
# sparse marix where s is support for k tuples
def pcy(D,s,k):
    rows = D.shape[0] # number of baskets
    cols = D.shape[1] # number of items
    buckets = np.zeros(
        shape=(cols//2), dtype=int)  
    bitmap = None

    # helper function increments buckets
    # given an array of hash values
    @np.vectorize
    def increment_buckets(index):
        buckets[index] += 1

    # loop through finding frequent 
    # k-tuples of support s
    for i in range(1, (2*k)):
        if (i % 2 == 1):
            j = (i // 2) + 2
            for r in range(rows):

                basket = D[r,:]
                basket = basket.toarray()
                basket = np.argwhere(basket == 1)[:,1]

                # generate canidate tuples
                canidates = itemcombos(basket,j)

                # hash each canidate tuple in canidates and increment bucket
                hash_tuple = lambda x: pcy_hash(len(buckets), x)
                hash_indices = np.apply_along_axis(hash_tuple,1,canidates)
                increment_buckets(hash_indices)
            bitmap = np.where(buckets > s,1,0)
        else:
            # TODO remove non frequent items
            pass


# TODO additional preprocessing dataset
# encode stocks to np.uint16 index
if __name__ == '__main__':
    # set support to 860 such that stocks are similar to
    # are 70 percent of the days in the last five years
    D = np.loadtxt('data/basketsenum.txt',dtype=int,delimiter=',')
    D = sp.csr_matrix(D)
    pcy(D,4,2)