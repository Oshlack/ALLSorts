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
import sys, os
from pathlib import Path

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def _getSubtypes(sub_hier, subtypes=[]):
    if sub_hier == False:  # Recursion stop condition
        return False

    for parent in sub_hier.keys():
        if isinstance(sub_hier[parent], dict):
            self._getSubtypes(sub_hier[parent], subtypes)
        else:
            subtypes.append(parent)

    return subtypes


def _flatHierarchy(sub_hier, flat_hierarchy={}):
    if sub_hier == False:  # Recursion stop condition
        return False

    for parent in sub_hier.keys():
        flat_hierarchy[parent] = _getSubtypes(sub_hier[parent], subtypes=[])
        _flatHierarchy(sub_hier[parent], flat_hierarchy=flat_hierarchy)

    return flat_hierarchy


def _pseudoCounts(counts, subtypes, name, parents, f_hierarchy):
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
    """Returns project root folder."""
    return Path(__file__).parent.parent


def message(message, level=False, important=False):

    text = "*** " + message + " ***" if important else message
    if level == 1:
        print("=======================================================================")
        print(text)
        print("=======================================================================")
    elif level == 2:
        print(text)
        print("-----------------------------------------------------------------------")
    else:
        print(text)

def root_dir() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent

def create_dir(path):

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