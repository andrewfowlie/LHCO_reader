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
such that the difference between the sums of the subsets is minimized.

In the non-standard problem, the function to be minimized is
arbitrary - it isn't simply the difference between the sums of the two subsets.
"""

###############################################################################

from __future__ import print_function
from __future__ import division

import copy
import sys
import warnings

from itertools import chain, combinations

###############################################################################

__author__ = "Andrew Fowlie"
__copyright__ = "Copyright 2015"
__credits__ = ["Luca Marzola"]
__license__ = "GPL"
__maintainer__ = "Andrew Fowlie"
__email__ = "Andrew.Fowlie@Monash.Edu.Au"
__status__ = "Production"

###############################################################################


def powerset(list_):
    """
    The set of all subsets, i.e. the powerset, of a set.
    See `itertools <https://docs.python.org/2/library/itertools.html>`_ and
    `wiki <https://en.wikipedia.org/wiki/Power_set>`_.

    .. warnings::
        A powerset of a set includes the empty subset and the set itself.

    :param list_: A list for which a powerset is desired
    :type list_: list

    :returns: Powerset of original list
    :rtype: itertools.chain

    :Example:

    >>> list(powerset([1, 2, 3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    combs = (combinations(list_, r) for r in range(len(list_) + 1))
    return chain.from_iterable(combs)

###############################################################################


def non_standard_brute(list_, measure_fn):
    """
    Solve a non-standard partition problem by brute force.

    :param list_: List of elements of any type
    :type list_: list
    :param measure_fn: Function for measure to be minimized for two subsums
    :type measure_fn: function

    :returns: Sums of two subsets, chosen such that measure minimized
    :rtype: list

    :Example:

    >>> non_standard_brute(example_list, example_fn)
    [194.31000000000003, 194.44999999999996]
    """
    min_measure = None
    for subset in powerset(list_):
        sum_1 = sum(subset)
        # Insure sum of empty subset is of correct type
        sum_1 += list_[0] * 0.
        sum_2 = sum(list_) - sum_1
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
    :param measure_fn: Function for measure to be minimized for two subsums
    :type measure_fn: function

    :returns: Sums of two subsets, chosen such that measure minimized
    :rtype: list

    :Example:

    >>> non_standard_greedy(example_list, example_fn)
    [195.31, 193.45000000000002]
    """

    measure = 0.
    # Initialize sums to zero, but of the correct type
    sum_1 = list_[0] * 0.
    sum_2 = list_[0] * 0.

    for item in list_:
        # Assign item to subset such that afffect on measure is minimal
        measure = measure_fn(sum_1, sum_2)
        diff_1 = measure_fn(sum_1 + item, sum_2) - measure
        diff_2 = measure_fn(sum_1, sum_2 + item) - measure
        if diff_1 < diff_2:
            sum_1 += item
        else:
            sum_2 += item

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
    internal_list = copy.deepcopy(list_)
    diff = 0.
    internal_list.sort(reverse=True)

    for item in internal_list:
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
    internal_list = copy.deepcopy(list_)
    while len(internal_list) > 1:
        internal_list.sort(reverse=True)
        diff = internal_list[0] - internal_list[1]
        del internal_list[0:2]
        internal_list.append(diff)

    diff = internal_list[0]
    return diff

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
    0.13999999999992951
    """
    internal_list = copy.deepcopy(list_)
    if len(internal_list) > 20:
        warnings.warn("Large powerset - consider an alternative strategy")

    sum_ = sum(internal_list)
    diff_subset = lambda subset: abs(sum_ - 2. * sum(subset))
    diff = min([diff_subset(subset) for subset in powerset(internal_list)])
    return diff

###############################################################################


def prune(CKK_branch):
    """
    Decorator that prunes search tree in the complete Karmarkar-Karp algorithm
    by stopping branches that cannot yield an improvement.

    :param CKK: A call for a branch in CKK algorithm
    :type branch: function
    """
    def pruner(list_, first=True):
        """
        Preliminary aspects of complete Karmarkar-Karp algorithm,
        including checking whether to prune a branch.

        :param list_: List of elements
        :type list_: list
        :param first: Whether first in recursive calls to CKK algorithm
        :type first: bool

        :returns: Result of :func:`branch` possibly halted
        :rtype: float

        :Example:

        >>> CKK(example_list)
        0.13999999999998414
        >>> CKK(example_list)
        0.13999999999998414
        """
        internal_list = copy.deepcopy(list_)

        # Reset minimum, if neccessary
        if first:
            pruner.min = float("inf")

        # Check whether end of branch or whether optimal solution achieved
        if pruner.min == 0.:
            return pruner.min
        elif len(internal_list) == 1:
            return list_[0]

        # Sort branch
        internal_list.sort(reverse=True)

        # Prune branch
        if pruner.min < internal_list[0] - sum(internal_list[1:]):
            return internal_list[0]

        # Find minimum in branch and track minimum of all branches
        min_branch = CKK_branch(internal_list)
        pruner.min = min(pruner.min, min_branch)

        return min_branch

    pruner.__name__ = CKK_branch.__name__
    pruner.min = float("inf")
    return pruner


@prune
def CKK(list_, first=True):
    """
    Divide elements into two approximately equal subsets
    with a complete Karmarkar-Karp algorithm.

    :param list_: A list of positive numbers
    :type list_: list
    :param first: Whether first in recursive calls to CKK algorithm
    :type first: bool

    :returns: Minimum difference between sums of subsets
    :rtype: float
    """
    internal_list = copy.deepcopy(list_)
    # Replace maximum two numbers by their difference and their sum in two
    # branches
    diff = internal_list[0] - internal_list[1]
    sum_ = internal_list[0] + internal_list[1]
    sum_tree = copy.deepcopy(internal_list)
    diff_tree = copy.deepcopy(internal_list)
    del sum_tree[0:2]
    del diff_tree[0:2]
    sum_tree.append(sum_)
    diff_tree.append(diff)

    return min(CKK(diff_tree, first=False), CKK(sum_tree, first=False))

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
    0.13999999999992951
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
    :param measure_fn: Function for measure to be minimized for two subsums
    :type measure_fn: function

    :returns: Sums of two subsets, chosen such that measure minimized
    :rtype: list

    :Example:

    >>> non_standard_solver(example_list, example_fn,
    ...                     algorithm="non_standard_brute")
    [194.31000000000003, 194.44999999999996]
    """
    this_module = sys.modules[__name__]
    try:
        diff = getattr(this_module, algorithm)(list_, measure_fn)
    except AttributeError:
        print("Unknown algorithm: %s" % algorithm)
        raise

    return diff

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
