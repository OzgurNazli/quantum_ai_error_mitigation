import functools
import itertools
import numpy as np
import torch
import cudaq
import matplotlib.pyplot as plt

import genQC
from genQC.imports import *
from genQC.pipeline.diffusion_pipeline import DiffusionPipeline
from genQC.inference.infer_compilation as infer_comp
from genQC.util as util 

#Fixed seed for reproducibility 
torch.manual_seed(0)
np.random.seed(0)

"""
The primary objective of a diffusionmodel is to generate high quality 
samples by learning to reverse a noise-adding process, rather than 
learning the data distribution directly.

Clean data set -> gaussian noise is incrementally added in a f-process
"""
"""
U-net architecture to learn the reverse process of denoising.
U-net is trainedtı take a noisy imagae as input and predct the specific
noise pattern that was added to it. 
min(loss) -> predicted noise VS actual noise
"""
# CNOT(q0,q3) ->2 (from q0 to q3 means that) (-2,0,0,2)
# H(q2)       ->1 (0,0,1,0)
# Tokenized matrix is number of qubits in cloumn, goin on time in a row

"""
For improved numerical stability during model training, the discrete 
tokenized matrix is embedded into a continuous tensor. The idea is to 
replace every integer in our matrix wit a unit vector chosen from a 
specifically prepared set of orthonoral basis vectors of dimension d.
"""
































