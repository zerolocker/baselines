"""
GFlag stands for "global flag".
This file is used for storing command line flags(i.e arguments)
so that they can be univerially accessible after importing this file.
"""
import sys

class GFlag(object):
  _dict = None # None denotes the uninitialized state

  def init_me_as(self, argsdict):
    if GFlag._dict is None:
      GFlag._dict = argsdict
    else:
      raise AttributeError("GFlag is already initialized")

  def __getattr__(self, name):
    if GFlag._dict is None:
      raise AttributeError("GFlag hasn't been initialized")
    if name not in GFlag._dict:
      raise AttributeError("Flag named '%s' is not found in GFlag" % name)
    return GFlag._dict[name]

  def __setattr__(self, name, val):
    raise AttributeError("GFlag is immutable after initialization")

# Replace the module with the object. See https://stackoverflow.com/questions/2447353/getattr-on-a-module
sys.modules[__name__] = GFlag()