# Core module initialization
import os

# Absolute path to the core/ package directory. Used by sub-modules
# (mof2zeo, simulation) to resolve data files relative to the package root
# instead of the current working directory.
__root_dir__ = os.path.dirname(os.path.abspath(__file__))
