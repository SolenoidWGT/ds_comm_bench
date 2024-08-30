'''Copyright The Microsoft DeepSpeed Team'''
from accelerator import accelerator

def get_accelerator():
    return accelerator

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'bfloat16'
DEFAULT_BACKEND = 'nccl'
DEFAULT_UNIT = 'GBps'
DEFAULT_DIST = 'torch'
DEFAULT_MAXSIZE = 26
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500
