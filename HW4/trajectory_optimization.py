import numpy as np

from copy import deepcopy
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

waypoints = 300
N = 101
OBST = np.array([[20, 30], [60, 40], [70, 85]])
epsilon = np.array([[25], [20], [30]])

obs_cost = np.zeros((N, N))
for i in range(OBST.shape[0]):
    t = np.ones((N, N))
    t[OBST[i, 0], OBST[i, 1]] = 0
    t_cost = distance_transform_edt(t)
    t_cost[t_cost > epsilon[i]] = epsilon[i]
    t_cost = 1 / (2 * epsilon[i]) * (t_cost - epsilon[i])**2
    obs_cost = obs_cost + t_cost

gx, gy = np.gradient(obs_cost)

SX = 10
SY = 10
GX = 90
GY = 90

traj = np.zeros((2, waypoints))
traj[0, 0] = SX
traj[1, 0] = SY
dist_x = GX-SX
dist_y = GY-SY
for i in range(1, waypoints):
    traj[0, i] = traj[0, i-1] + dist_x/(waypoints-1)
    traj[1, i] = traj[1, i-1] + dist_y/(waypoints-1)

def savefig(path, name):
    plt.clf()
    #print(path)
    tt = path.shape[0]
    path_values = np.zeros((tt, 1))
    for i in range(tt):
        path_values[i] = obs_cost[int(np.floor(path[i, 0])), int(np.floor(path[i, 1]))]

    # Plot 2D
    plt.imshow(obs_cost.T)
    plt.plot(path[:, 0], path[:, 1], 'ro')
    plt.savefig("figures/" + name + "_2D.png")

    # Plot 3D
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(range(N), range(N))
    ax3d.plot_surface(xx, yy, obs_cost.T, cmap=plt.get_cmap('coolwarm'))
    ax3d.scatter(path[:, 0], path[:, 1], path_values, s=20, c='r')

    plt.show()
    plt.savefig("figures/" + name + "_3D.png")

path_init = traj.T
savefig(path_init, "original")

interp_gx = LinearNDInterpolator([(i // N, i % N) for i in range(N**2)], [gx[i // N, i % N] for i in range(N**2)])
interp_gy = LinearNDInterpolator([(i // N, i % N) for i in range(N**2)], [gy[i // N, i % N] for i in range(N**2)])


# Part a
def optimize_a_step(traj, lr=0.1):
    grad = np.zeros_like(traj)
    subset = traj[1 : waypoints - 1]
    #print(subset)
    subset_gradx = interp_gx(subset)
    subset_grady = interp_gy(subset)
    #print(subset_gradx)
    #print(gx[10, 10], gx[11, 11])
    grad[1 : waypoints - 1, 0] += subset_gradx
    grad[1 : waypoints - 1, 1] += subset_grady
    traj -= lr * grad
    return traj, grad

traj = deepcopy(path_init)
traj, grad = optimize_a_step(traj)
savefig(traj, "onestep_a")

i = 1
while np.sum(np.sqrt(np.power(grad, 2))) > 1:
    i += 1
    #print(np.sum(np.sqrt(np.power(grad, 2))))
    traj, grad = optimize_a_step(traj)
    traj = np.clip(traj, 0, 100)
    #print(i)

print("Converged after {} iterations".format(i))
savefig(traj, "convergence_a")

# Part b
def optimize_b_step(traj, lr=0.1):
    grad = np.zeros_like(traj)
    subset = traj[1 : waypoints - 1]
    #print(subset)
    subset_gradx = interp_gx(subset)
    subset_grady = interp_gy(subset)

    grad[1 : waypoints - 1, 0] += 0.8 * subset_gradx
    grad[1 : waypoints - 1, 1] += 0.8 * subset_grady

    for i in range(1, waypoints - 1):
        p_i = traj[i]
        p_im1 = traj[i - 1]
        grad[i, 0] += 4 * (p_i[0] - p_im1[0])
        grad[i, 1] += 4 * (p_i[1] - p_im1[1])
    #print(subset_gradx)
    #print(gx[10, 10], gx[11, 11])
    traj -= lr * grad
    return traj, grad

traj = deepcopy(path_init)
traj, grad = optimize_b_step(traj)
savefig(traj, "onestep_b")

i = 1
while i < 500:
    i += 1
    traj, grad = optimize_b_step(traj)
    traj = np.clip(traj, 0, 100)
    if i == 100:
        savefig(traj, "100_b")
    elif i == 200:
        savefig(traj, "200_b")
    elif i == 500:
        savefig(traj, "500_b")
    #print(i)

# Part c
def optimize_c_step(traj, lr=0.1):
    grad = np.zeros_like(traj)
    subset = traj[1 : waypoints - 1]
    #print(subset)
    subset_gradx = interp_gx(subset)
    subset_grady = interp_gy(subset)

    grad[1 : waypoints - 1, 0] += 0.8 * subset_gradx
    grad[1 : waypoints - 1, 1] += 0.8 * subset_grady

    for i in range(1, waypoints - 1):
        p_i = traj[i]
        p_im1 = traj[i - 1]
        p_ip1 = traj[i + 1]
        grad[i, 0] += 4 * (2 * p_i[0] - p_im1[0] - p_ip1[0])
        grad[i, 1] += 4 * (2 * p_i[1] - p_im1[1] - p_ip1[1])
    #print(subset_gradx)
    #print(gx[10, 10], gx[11, 11])
    traj -= lr * grad
    return traj, grad

traj = deepcopy(path_init)

i = 0
while i < 1000:
    i += 1
    traj, grad = optimize_c_step(traj)
    traj = np.clip(traj, 0, 100)
    if i == 1000:
        savefig(traj, "1000_c")
    elif i == 500:
        savefig(traj, "500_c")
    #print(i)
