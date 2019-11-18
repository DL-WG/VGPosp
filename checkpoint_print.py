import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import sys
from tensorflow.python.tools import inspect_checkpoint as chkp


def chkp_print(saver_path):
    chkp.print_tensors_in_checkpoint_file(saver_path, tensor_name='', all_tensors=True)
    pass

if __name__ == '__main__':
    chkp_print(sys.argv[1])
    pass