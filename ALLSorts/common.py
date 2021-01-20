#=======================================================================================================================
#
#   ALLSorts v2 - Common functions
#   Author: Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' External '''
import os
from pathlib import Path

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def _flat_hierarchy(sub_hier, flat_hierarchy={}):

    """
    A recursive function (I'm sorry) that converts a dictionary representing a hierarchy into a single level.

    ...

    Parameters
    __________
    sub_hier : Dict
        Hierarchical dictionary (a branch in the hierarchy)
    flat_hierarchy : Dict
        Flat dictionary recursively built by navigating through each branch in the hierarchy.

    Returns
    __________
    flat_hierarchy : Dict
        Flat dictionary recursively built by navigating through each branch in the hierarchy.
    """

    if sub_hier == False:  # Recursion stop condition
        return False

    for parent in sub_hier.keys():
        flat_hierarchy[parent] = _get_subtypes(sub_hier[parent], subtypes=[])
        _flat_hierarchy(sub_hier[parent], flat_hierarchy=flat_hierarchy)

    return flat_hierarchy

def _get_subtypes(sub_hier, subtypes=[]):

    """
    A recursive function (I'm sorry) that converts lists all subtypes at the terminal nodes of a hierarchical branch.

    ...

    Parameters
    __________
    sub_hier : Dict
        Hierarchical dictionary (a branch in the hierarchy)
    subtypes : List
        A list of all leaf nodes within the supplied hierarchical branch.

    Returns
    __________
    subtypes : List
        A list of all leaf nodes within the supplied hierarchical branch.
    """

    if sub_hier == False:  # Recursion stop condition
        return False

    for parent in sub_hier.keys():
        if isinstance(sub_hier[parent], dict):
            _get_subtypes(sub_hier[parent], subtypes)
        else:
            subtypes.append(parent)

    return subtypes

def _pseudo_counts(counts, subtypes, parents, f_hierarchy):

    """
    Create a subset of the counts that reflects the classification task at this level in the hierarchy.

    I.e. Ph-like vs. Ph classification does not need samples from TCF3-PBX1.
         Ph Group vs. Rest does not need samples to be labelled as Ph or Ph-like (but Ph Group)

    ...

    Parameters
    __________
    counts : Pandas DataFrame
        Pandas DataFrame that represents the samples (rows) x genes (columns)).
    subtypes : List
        A list of all leaf nodes within the supplied hierarchical branch.
    parents : List
        A list of the parent nodes within this branch of the hierarchy.
    f_hierarchy : dict
        A flattened hierarchy

    Returns
    __________
    p_counts : Pandas DataFrame
        Subset of supplied counts that represent the classification problem at this node in the hierarchy.
    p_subtypes : Pandas Series
        Modified labels for the subset of counts at this level
    """

    p_counts = counts.copy()
    p_subtypes = subtypes.copy()

    # Create the psuedo subtypes
    for parent in parents:
        if parent in f_hierarchy:
            children = f_hierarchy[parent]
            if children:
                p_subtypes[p_subtypes.isin(children)] = parent

    # Dump any samples that are not needed for this level
    dump = p_subtypes[~p_subtypes.isin(parents)].index
    p_counts.drop(dump, inplace=True)
    p_subtypes.drop(dump, inplace=True)

    return p_counts, p_subtypes

def get_project_root() -> Path:

    """
    Pretty straightforward - Return the project root directory path relative to this file.

    ...

    Returns
    __________
    str
        Path to root directory
    """

    return Path(__file__).parent.parent

def message(message, level=False, important=False):

    """
    A simple way to print a message to the user with some formatting.

    ...

    Output
    __________
    A stylish, printed message.
    """

    text = "*** " + message + " ***" if important else message
    if level == 1:
        print("=======================================================================")
        print(text)
        print("=======================================================================")
    elif level == 2:
        print(text)
        print("-----------------------------------------------------------------------")
    elif level == "w":
        print("\n***********************************************************************")
        print(text)
        print("***********************************************************************\n")

    else:
        print(text)

def root_dir() -> Path:

    """
    Pretty straightforward - Return the parent directory path relative to this file.

    ...

    Returns
    __________
    str
        Path to root directory
    """

    return Path(__file__).parent

def create_dir(path):

    """
    Create a directory given the supplied path.

    ...

    Output
    __________
    Directory created in the system
    """

    if isinstance(path, list):
        for p in path:
            try:
                os.mkdir(p)
            except OSError:
                continue
    else:
        try:
            os.mkdir(path)
        except OSError:
            pass