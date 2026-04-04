import os

__version__ = "0.0.0"
# __root_dir__ is the parent of mof2zeo folder (i.e., core/)
# So we need to go up one level from mof2zeo/
__root_dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
