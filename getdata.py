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

##########################################################
####################### Mesh #############################
##########################################################
maxV = 1e-3
x_min, x_max, y_min, y_max = -1.0, 1.5, -1.0, 1.0
Lx = x_max - x_min
Ly = y_max - y_min
m = max(Lx / Ly, Ly / Lx)
N_maille_x = 2 * int(np.floor(np.sqrt(1/maxV)) * Lx / m )
N_maille_y = 2 * int(np.floor(np.sqrt(1/maxV)) * Ly / m )

points = np.array([
    [x_min, y_min],
    [x_max, y_min],
    [x_max, y_max],
    [x_min, y_max]
])
markers  = [4,4,4,4]
facets = round_trip_connect(0, len(points)-1)

# data NACA00012
with open('data/NACA012/NACA012.txt', 'r') as f:
    data = f.read()

data = data.splitlines()
data = [line.split() for line in data]
data = np.array(data, dtype=float)


outter_start = len(points)
points = np.vstack((points, data))
facets.extend(round_trip_connect(outter_start, len(points) - 1))
markers.extend([2] * (len(data) - 1))

def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i+1))
    result.append((end, start))
    return result

info = triangle.MeshInfo()
info.set_points(points)
info.set_facets(facets, facet_markers=markers)
info.set_holes([(0.5, 0.)])

Mesh = mesh.Mesh()
Mesh.mesh_generator(info, maxV=maxV, min_angle=30.0)
ids = jnp.where(Mesh.face_markers > 5)[0]
Mesh.face_markers = Mesh.face_markers.at[ids].set(2)
Mesh.plot_mesh()


##########################################################
######################### IC #############################
##########################################################

Prim = jnp.array([1.0, 2.0, 0.0, 1.4])

N_s = 50
Snapshots = jnp.zeros((N_s, len(Mesh.tris), 4))

for i, M in enumerate(jnp.linspace(1.3,2.5,N_s)):
    print("snapshot : ", i, " / ", N_s)
    Prim = jnp.array([1.0, M, 0.0, 1.4])

    Primitives = jnp.zeros((len(Mesh.tris), 4))
    Primitives = Primitives.at[:, :].set(Prim)
    W = helper.getConserved(Primitives, gamma=1.4)
    Snapshots = Snapshots.at[i].set(W)


##########################################################
####################### solver #############################
##########################################################

t_final = 2.
dt = helper.get_dt(W, Mesh, CFL = 0.3)
N_t = int(t_final / dt) + 1

start_time = time.time()

kwargs = {'gamma': 1.4,
    'mu': 1.716e-4,
    'R': 287,
    'k': 0.30780,
    'flag_NS': False,
    'alpha': 1.}

integration_fn = partial(TI.time_step_RK2, dt=dt, mesh=Mesh, residual=Euler.residual, **kwargs)
vmap_step = jax.vmap(integration_fn)

for n in range(N_t):
    Snapshots = vmap_step(Snapshots)
    if n % 1000 == 0:
        print(f'It : {n} / {N_t}')



path = "data/NACA012/"
jnp.save(path + "snapshot_matrix.npy", Snapshots)
jnp.save(path + "mesh.npy", Mesh)

jnp.isnan(Snapshots).any()

Mesh.plot_solution(Snapshots[0, :, 0], labels = r'$\rho$')