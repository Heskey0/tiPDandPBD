import taichi as ti
import taichi.math as tm
from scipy.sparse import dia_matrix, csc_matrix, linalg
from tiReadMesh import *

# ti.init()

numSubsteps = 10
jacobi_iter = 10
jacobi_alpha = 0.1
stiffness = 5000
h = 1.0 / 60.0 / numSubsteps
h_inv = 1.0 / h

surf_show = ti.field(int, numSurfs * 3)
surf_show.from_numpy(surf_np.flatten())


# initialize

N = numParticles
N_tetrahedron = numTets
dim = 3*N

Dm = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)

x_proj = ti.Vector.field(3, ti.f32, N)
x_iter = ti.Vector.field(3, ti.f32, N)
count = ti.field(ti.i32, N)

prevPos = ti.Vector.field(3, ti.f32, numParticles)
vel = ti.Vector.field(3, ti.f32, N)
f_ext = ti.Vector.field(3, ti.f32, N)

# ---------------------------------------------------------------------------- #
#                                      Init                                    #
# ---------------------------------------------------------------------------- #

# init position and velocity
def initialize():
    for index in range(N):
        x_proj[index] = ti.Vector([0.0, 0.0, 0.0])
        vel[index] = ti.Vector([0.0, 0.0, 0.0])
        f_ext[index] = ti.Vector([0.0, -9.8 * 1.0, 0.0])


def initialize_M():
    data = np.ones(dim)
    offset = np.array([0])
    return dia_matrix((data, offset), shape=(dim, dim))

M = initialize_M()


def initialize_Si():
    data = np.ones(12)  # set A_i=B_i=I
    GcT = []
    sum_SiTSi_tmp = csc_matrix((dim, dim))
    for i in range(N_tetrahedron):
        row = np.arange(12)
        col = []
        for n in range(4):
            ind = tet[i][n]
            col.append(3 * ind + 0)
            col.append(3 * ind + 1)
            col.append(3 * ind + 2)
        col_nd = np.array(col)
        Si = csc_matrix((data, (row, col_nd)), shape=(12, dim))
        SiT = Si.transpose()
        SiTSi = SiT @ Si  # diagonal matrix
        GcT.append(SiT)
        sum_SiTSi_tmp = sum_SiTSi_tmp + SiTSi

    return GcT, sum_SiTSi_tmp

SiT, sum_SiTSi = initialize_Si()


# ---------------------------------------------------------------------------- #
#                                      Core                                    #
# ---------------------------------------------------------------------------- #
@ti.func
def compute_D(i):
    q = tet[i][0]
    w = tet[i][1]
    e = tet[i][2]
    r = tet[i][3]
    return ti.Matrix.cols(
        [pos[q] - pos[r], pos[w] - pos[r], pos[e] - pos[r]])

@ti.func
def compute_D_local(i):
    q = tet[i][0]
    w = tet[i][1]
    e = tet[i][2]
    r = tet[i][3]
    return ti.Matrix.cols(
        [x_proj[q] - x_proj[r], x_proj[w] - x_proj[r], x_proj[e] - x_proj[r]])

@ti.func
def compute_F(i):
    return compute_D_local(i) @ Dm_inv[i]


@ti.kernel
def initialize_Dm():
    for i in range(N_tetrahedron):
        Dm[i] = compute_D(i)
        Dm_inv[i] = Dm[i].inverse()


initialize()
initialize_Dm()


@ti.func
def jacobi():
    for it in range(N_tetrahedron):
        F = compute_F(it)
        U, S, V = ti.svd(F)
        S[0, 0] = min(max(0.95, S[0, 0]), 1.05)
        S[1, 1] = min(max(0.95, S[1, 1]), 1.05)
        S[2, 2] = min(max(0.95, S[2, 2]), 1.05)
        F_star = U @ S @ V.transpose()
        D_star = F_star @ Dm[it]

        q = tet[it][0]
        w = tet[it][1]
        e = tet[it][2]
        r = tet[it][3]

        # find the center of gravity
        center = (x_proj[q] + x_proj[w] + x_proj[e] + x_proj[r]) / 4

        # find the projected vector
        for n in ti.static(range(3)):
            x3_new = center[n] - (D_star[n, 0] + D_star[n, 1] + D_star[n, 2]) / 4
            x_iter[r][n] += x3_new
            x_iter[q][n] += x3_new + D_star[n, 0]
            x_iter[w][n] += x3_new + D_star[n, 1]
            x_iter[e][n] += x3_new + D_star[n, 2]

        count[q] += 1
        count[w] += 1
        count[e] += 1
        count[r] += 1

    for i in range(N):
        x_proj[i] = (x_iter[i] + jacobi_alpha * x_proj[i]) / (count[i] + jacobi_alpha)


@ti.kernel
def local_step():
    for i in range(N):
        x_proj[i] = pos[i]
        x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
        count[i] = 0
    # Jacobi Solver
    for k in range(jacobi_iter):
        for i in range(N):
            x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
            count[i] = 0
        jacobi()


def global_step(sn):
    h2_inv = h_inv ** 2

    xn = pos.to_numpy().reshape(dim)
    p = x_proj.to_numpy().reshape(dim)

    A = (h_inv ** 2) * M + stiffness * sum_SiTSi
    b = h2_inv * M @ sn + stiffness * sum_SiTSi @ p
    x_star, info = linalg.cg(A, b, x0=xn)

    return x_star


@ti.kernel
def updatePos(x_star: ti.types.ndarray()):
    for i in range(N):
        x_new = ti.Vector([x_star[3 * i + 0], x_star[3 * i + 1], x_star[3 * i + 2]])
        pos[i] = x_new
        if x_new[1] < 0.1:
            pos[i][1] = 0.1


@ti.kernel
def updateVel():
    for i in range(N):
        vel[i] = (pos[i] - prevPos[i]) * h_inv
        if pos[i][1] < 0.1:
            pos[i][1] = 0.1
            if vel[i][1] < 0:
                vel[i][1] = 0.0

@ti.kernel
def initialize_solution(sn: ti.types.ndarray()):
    for i in range(N):
        pos[i] = ti.Vector([sn[3*i+0], sn[3*i+1], sn[3*i+2]])


def step():
    xn = pos.to_numpy().reshape(dim)
    vn = vel.to_numpy().reshape(dim)
    prevPos.copy_from(pos)
    f_ext_n = f_ext.to_numpy().reshape(dim)

    # Projective Dynamics
    sn = xn + h * vn + (h ** 2) * linalg.inv(M) @ f_ext_n

    initialize_solution(sn)
    for i in range(numSubsteps):
        local_step()
        x_star = global_step(sn)
        updatePos(x_star)
    updateVel()


# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
# init the window, canvas, scene and camerea
window = ti.ui.Window("Projective Dynamics", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.067, 0.184, 0.255))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# initial camera position
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


@ti.kernel
def init_pos():
    for i in range(numParticles):
        pos[i] += tm.vec3(0.5, 1, 0)


def main():
    init_pos()
    while window.running:
        # do the simulation in each step
        step()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(pos, indices=surf_show, color=(1, 1, 0))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()