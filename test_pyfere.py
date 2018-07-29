import pyfere as pf
import numpy as np
import time

def execute(BloomFilter, n, m, k, in_parallel=True):

    def get_data(n):
        return list(np.random.choice(2**63-1, n))

    def benchmark(func, tag):

        start = time.time()
        result = func()
        end = time.time()

        print ("%ss (%s)" % (end-start, tag))

        return result

    Filter = BloomFilter(m, k, in_parallel)

    print ("Benchmark %s" % Filter, n, m, k)

    x = get_data(n)
    y = get_data(n)

    while len(set(x) & set(y)) > 0 or \
          len(set(x)) < n or len(set(y)) < n:
        x = get_data(n)
        y = get_data(n)

    op_insert = lambda : Filter.insert(x)
    benchmark(op_insert, "insert x")

    op_query = lambda : Filter.query(x)
    result = benchmark(op_query, "query x")

    print ("true positive rate: TPR = %s" % (sum(result)/float(n)))

    op_query = lambda : Filter.query(y)
    result = benchmark(op_query, "query y")

    print ("false positive rate: FPR = %s" % (sum(result)/float(n)))

n = 10**7
m = 8*n
k = int(float(m)/float(n)*np.log(2))

execute(pf.BloomFilter, n, m, k)
execute(pf.PartitionedBloomFilter, n, m, k)
