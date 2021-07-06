from collections import Counter

cimport cython
import numpy as np
cimport numpy as np
import time

cdef char_pair(a, b):
    if a < b:
        return '' + a + b
    else:
        return '' + b + a


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sw_align(a, text, b_start, b_end, gap_score, match_score, mismatch_score):
    b = text[b_start:b_end]
    cdef int n = len(a)
    cdef int m = len(b)
    start_time = time.time()
    cdef np.ndarray[np.int32_t, ndim=2] f = np.empty((n + 1, m + 1), np.int32)
    f[0,0] = 0
    cdef int i = 0
    cdef int j = 0
    for i in range(1, n + 1):
        f[i,0] = gap_score * i
    for j in range(1, m + 1):
        f[0,j] = gap_score * j
    cdef int max_score = 0
    cdef int start_i = 0
    cdef int start_j = 0
    cdef int match = 0
    cdef int insert = 0
    cdef int delete = 0
    cdef int score = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = f[i - 1,j - 1] + (match_score if a[i-1] == b[j-1] 
                                       else mismatch_score)
            insert = f[i,j - 1] + gap_score
            delete = f[i - 1,j] + gap_score
            score = max(0, match, insert, delete)
            # print(f"{i} {j} {score}")
            f[i,j] = score
            if score > max_score:
                max_score = score
                start_i = i
                start_j = j
    end_time = time.time()
    print(f"sw_align_sped_up compute matrix={end_time - start_time}")

    start_time = time.time()
    substitutions = Counter()
    i = start_i
    j = start_j
    while (j > 0 or i > 0) and f[i,j] != 0:
        if i > 0 and j > 0 and f[i,j] == (f[i-1,j-1] + (match_score if a[i-1] == b[j-1] 
                                                        else mismatch_score)):
            substitutions[char_pair(a[i-1], b[j-1])] += 1
            i -= 1
            j -= 1
        elif i > 0 and f[i,j] == (f[i-1,j] + gap_score):
            i -= 1
        elif j > 0 and f[i,j] == (f[i,j-1] + gap_score):
            j -= 1
        else:
            raise Exception('Switch-Waterman failure')
    cdef int align_start = max(b_start, b_start + j - 1)
    cdef int align_end = min(b_end, b_start + start_j)
    cdef double final_score = f[start_i, start_j] / (match_score * max(align_end - align_start, n))
    end_time = time.time()
    print(f"sw_align_sped_up backtrack={end_time - start_time}")
    return align_start, align_end, final_score, substitutions
