import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    print(pd.__version__)
