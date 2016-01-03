#!/usr/bin/env python
r"""
=================
Partition problem
=================

Functions to solve the
`partition problem <https://en.wikipedia.org/wiki/Partition_problem>`_
and non-standard partition problems.

We implement greedy, brute force and Karmarkar-Karp algorithms.

In the standard problem, we attempt to divide a set into two subsets,
such that the difference between the sums of the subsets is minimised.

In the non-standard problem, the function to be minimised is
arbitrary - it isn't simply the difference between the sums of the two subsets.
"""

###############################################################################

from __future__ import print_function
from __future__ import division

import sys
import warnings

from itertools import chain, combinations
from bisect import insort

###############################################################################

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2015"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__maintainer__ = "Andrew Fowlie"
__email__ = "Andrew.Fowlie@Monash.Edu.Au"
__status__ = "Production"

###############################################################################


def powerset(list_, half=False):
    """
    The set of all subsets, i.e. the powerset, of a set.
    See `itertools <https://docs.python.org/2/library/itertools.html>`_ and
    `wiki <https://en.wikipedia.org/wiki/Power_set>`_.

    .. warnings::
        A powerset of a set includes the empty subset and the set itself.

    :param list_: A list for which a powerset is desired
    :type list_: list
    :param half: Build half the powerset - the missing half is contained in \
    the complement
    :type half: bool

    :returns: Powerset of original list
    :rtype: itertools.chain

    :Example:

    >>> list(powerset([1, 2, 3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    >>> list(powerset([1, 2, 3], half=True))
    [(), (1,), (2,), (3,)]
    """
    if half:
        max_r = int(len(list_) / 2)  # Round down
    else:
        max_r = len(list_)
    combs = (combinations(list_, r) for r in range(max_r + 1))
    return chain.from_iterable(combs)

###############################################################################


def non_standard_brute(list_, measure_fn):
    """
    Solve a non-standard partition problem by brute force.

    :param list_: List of elements of any type
    :type list_: list
    :param measure_fn: Function for measure to be minimised for two sub-sums
    :type measure_fn: function

    :returns: Sums of two subsets, chosen such that measure minimised
    :rtype: list

    :Example:

    >>> non_standard_brute(example_list, example_fn)
    [194.45, 194.31]
    """
    sum_list = sum(list_)
    min_measure = None
    for subset in powerset(list_, half=True):
        sum_1 = sum(subset)
        sum_2 = sum_list - sum_1
        measure = measure_fn(sum_1, sum_2)
        if min_measure is None or measure < min_measure:
            min_measure = measure
            sums = [sum_1, sum_2]

    return sums

###############################################################################


def non_standard_greedy(list_, measure_fn):
    """
    Solve a non-standard partition problem with a "greedy" algorithm.

    .. warning::
        A "greedy" algorithm may not return the optimal solution.

    :param list_: List of elements of any type
    :type list_: list
    :param measure_fn: Function for measure to be minimised for two sub-sums
    :type measure_fn: function

    :returns: Sums of two subsets, chosen such that measure minimised
    :rtype: list

    :Example:

    >>> non_standard_greedy(example_list, example_fn)
    [195.31, 193.45000000000002]
    """

    sum_1 = 0.
    sum_2 = 0.
    for item in list_:
        # Assign item to subset such that affect on measure is minimal
        trial_1 = sum_1 + item
        measure_1 = measure_fn(trial_1, sum_2)
        trial_2 = sum_2 + item
        measure_2 = measure_fn(sum_1, trial_2)
        if measure_1 < measure_2:
            sum_1 = trial_1
        else:
            sum_2 = trial_2

    sums = [sum_1, sum_2]
    return sums

###############################################################################


def greedy(list_):
    """
    Divide elements into two approximately equal subsets
    with a "greedy" algorithm. Order the elements then assign elements
    to the smallest subset.

    .. warning::
        A "greedy" algorithm may not return the optimal solution.

    :param list_: A list of positive numbers
    :type list_: list

    :returns: Minimum difference between sums of subsets
    :rtype: float

    :Example:

    >>> greedy(example_list)
    8.46
    """
    list_ = sorted(list_, reverse=True)
    diff = 0.
    for item in list_:
        if diff <= 0.:
            diff += item
        else:
            diff -= item

    return abs(diff)

###############################################################################


def KK(list_):
    """
    Divide elements into two approximately equal subsets
    with the Karmarkar-Karp algorithm also known as the
    Least Difference Method.

    .. warning::
        This algorithm may not return the optimal solution.

    :param list_: A list of positive numbers
    :type list_: list

    :returns: Minimum difference between sums of subsets
    :rtype: float

    :Example:

    >>> KK(example_list)
    0.14000000000001078
    """
    list_ = sorted(list_)
    while len(list_) > 1:
        diff = list_.pop() - list_.pop()
        insort(list_, diff)

    return list_[0]

###############################################################################


def brute(list_):
    r"""
    Divide elements into two approximately equal subsets
    with a brute force algorithm.

    .. warning::
        This is complexity :math:`\mathcal{O}(2^n)` - this could be very slow.

    :param list_: A list of positive numbers
    :type list_: list

    :returns: Minimum difference between sums of subsets
    :rtype: float

    >>> brute(example_list)
    0.13999999999998636
    """
    if len(list_) > 20:
        warnings.warn("Large powerset - consider an alternative strategy")

    sum_ = sum(list_)
    diff_subset = lambda subset: abs(sum_ - 2. * sum(subset))
    diff_set = map(diff_subset, powerset(list_, half=True))
    diff = min(diff_set)
    return diff

###############################################################################


def prune(CKK_branch):
    """
    Decorator that prunes search tree in the complete Karmarkar-Karp algorithm
    by stopping branches that cannot yield an improvement.

    :param CKK: A call for a branch in CKK algorithm
    :type branch: function
    """
    def pruner(list_, branch=False):
        """
        Preliminary aspects of complete Karmarkar-Karp algorithm,
        including checking whether to prune a branch.

        :param list_: List of elements
        :type list_: list
        :param branch: Whether a branch in a CKK search
        :type branch: bool

        :returns: Result of :func:`branch` possibly halted
        :rtype: float

        :Example:

        >>> CKK(example_list)
        0.13999999999998414
        >>> CKK(example_list)
        0.13999999999998414
        """
        if len(list_) == 1:

            return list_[-1]

        if not branch:

            # Sort branch and reset minimum, if necessary
            list_ = sorted(list_)
            pruner.min = float("inf")
            return CKK_branch(list_)

        elif pruner.min < list_[-1] - sum(list_[:-1]):

            # Prune branch
            return pruner.min

        elif pruner.min == 0.:

            # Optimum solution reached
            return pruner.min

        else:
            # Find minimum in branch and track minimum of all branches
            min_branch = CKK_branch(list_)
            pruner.min = min(pruner.min, min_branch)
            return min_branch

    pruner.__name__ = CKK_branch.__name__
    return pruner


@prune
def CKK(list_):
    """
    Divide elements into two approximately equal subsets
    with a complete Karmarkar-Karp algorithm.

    :param list_: A list of positive numbers
    :type list_: list

    :returns: Minimum difference between sums of subsets
    :rtype: float
    """
    # Replace maximum two numbers by their difference and their sum in two
    # branches
    diff = list_[-1] - list_[-2]
    sum_ = list_[-1] + list_[-2]
    del list_[-2:]
    sum_tree = list_ + [sum_]
    diff_tree = list_[:]
    insort(diff_tree, diff)

    return min(CKK(diff_tree, branch=True), CKK(sum_tree, branch=True))

###############################################################################


def solver(list_, algorithm="CKK"):
    """
    Wrapper for possible partition solving algorithms.

    :param algorithm: Choice of algorithm
    :type algorithm: string

    :returns: Minimum difference between sums of subsets
    :rtype: float

    :Example:

    >>> solver(example_list, algorithm="brute")
    0.13999999999998636
    """
    this_module = sys.modules[__name__]
    try:
        diff = getattr(this_module, algorithm)(list_)
    except AttributeError:
        print("Unknown algorithm: %s" % algorithm)
        raise

    return diff

###############################################################################


def non_standard_solver(list_, measure_fn, algorithm="non_standard_brute"):
    """
    Wrapper for possible non-standard partition solving algorithms.

    :param algorithm: Choice of algorithm
    :type algorithm: string
    :param measure_fn: Function for measure to be minimised for two sub-sums
    :type measure_fn: function

    :returns: Sums of two subsets, chosen such that measure minimised
    :rtype: list

    :Example:

    >>> non_standard_solver(example_list, example_fn,
    ...                     algorithm="non_standard_brute")
    [194.45, 194.31]
    """
    this_module = sys.modules[__name__]
    try:
        sums = getattr(this_module, algorithm)(list_, measure_fn)
    except AttributeError:
        print("Unknown algorithm: %s" % algorithm)
        raise

    return sums

###############################################################################

if __name__ == "__main__":
    import doctest
    example_list = [1.4,
                    10.1,
                    19.55,
                    11.71,
                    51.7,
                    122.1,
                    11.9,
                    25.1,
                    13.2,
                    22.7,
                    51.4,
                    37.8,
                    10.1
                    ]
    example_fn = lambda sum_1, sum_2: abs(sum_1 - sum_2)
    doctest.testmod(extraglobs={'test': example_list,
                                'example_fn': example_fn
                                })
