from functools import partial
import os
import time

from jax import config
import sys
sys.path.append('..') 
import jax
import jax.numpy as jnp
import jax_fvm.src.mesh.mesh as mesh
import jax_fvm.src.solvers.helper as helper
import jax_fvm.src.solvers.Time_Integration as TI
import jax_fvm.src.solvers.Euler.Euler as Euler
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import meshpy.triangle as triangle
import matplotlib.pyplot as plt
size = 14
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)

def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i+1))
    result.append((end, start))
    return result



path = "data/NACA012/"
Snapshots = jnp.load(path + "snapshot_matrix.npy")
Mesh = jnp.load(path + "mesh.npy" , allow_pickle = True).item()

Mesh.plot_solution(Snapshots[0, :, 0], labels = r'$\rho$')

# for i in range(Snapshots.shape[0]):
#     Mesh.plot_solution(Snapshots[i, :, 0], labels = r'$\rho$')