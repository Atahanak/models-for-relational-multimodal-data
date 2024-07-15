import sys
import os
module_path = os.path.join(os.path.dirname(__file__), 'build')
if module_path not in sys.path:
    sys.path.append(module_path)
#from build import negative_samples
import negative_sampling

__all__ = [
    'negative_sampling'
]