import numpy as np
import config
import utils
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import healpy as hp


cpt = 0
nside = 1
Npix = 12*nside**2
lmax = 2*nside
noise = 40**2*(4*np.pi)/Npix
noise_covar = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
#noise = 2
#noise_covar = np.ones(len(noise_covar))*2

fwhm_radians = (np.pi / 180) * 0.35
bl_gauss = hp.gauss_beam(fwhm_radians, lmax=lmax)[2:]
beam = np.array([bl for l, bl in enumerate(bl_gauss, start=2) for i in range(2 * l + 1)])
#beam = np.ones(len(beam))


theta_true = config.COSMO_PARAMS_MEAN + config.COSMO_PARAMS_SIGMA
cls_true = utils.generate_cls(theta_true)[:lmax-1]
var_cl_full = utils.generate_var_cls(cls_true)
var_cls = var_cl_full

alm_true = np.random.normal(scale=np.sqrt(var_cl_full))
pix_map = beam*alm_true + np.random.normal(scale=np.sqrt(noise_covar))
#pix_map = np.zeros(len(pix_map))
#pix_map = np.zeros(len(var_cl_full))
#var_cl_full = np.ones(len(var_cl_full))*2
#var_cls =np.ones(len(beam))*2
#pix_map = np.zeros(len(var_cls))

sigma_real = 1/(beam**2/noise_covar + 1/var_cls)
mu = sigma_real*beam*(1/noise_covar)*pix_map

print("Sigma real")
plt.plot(sigma_real)
plt.show()
print("mean")
print(mu[cpt])

plt.plot(1/sigma_real)
plt.show()

def compute_energy(x):
    precision = beam**2/noise_covar + 1/var_cls
    r = beam*(1/noise_covar)*pix_map
    return (1/2) * np.sum(x ** 2*precision) - np.sum(x*r)

def compute_gradient_energy(x):
    precision = beam ** 2 / noise_covar + 1 / var_cls
    r = beam*(1/noise_covar)*pix_map
    return precision*x - r

def bounce(position, velocity):
    grad = compute_gradient_energy(position)
    dot_prod = np.sum(grad * velocity)
    norm_squared_grad = np.sum(grad ** 2)
    new_velocity = velocity - 2 * (dot_prod / norm_squared_grad) * grad
    return new_velocity

def minimize_along_traj(position, velocity):
    precision = beam**2/noise_covar + 1/var_cls
    r = beam * (1 / noise_covar) * pix_map
    tau_star = (np.sum(velocity*r)-np.sum(position*velocity*precision))/np.sum((velocity**2*precision))
    tau_star = max(tau_star, 0)
    return tau_star

def solve_for_time(position, velocity, tau_star):
    precision = beam**2/noise_covar + 1/var_cls
    r = beam * (1 / noise_covar) * pix_map
    V = np.random.uniform()
    u_star = compute_energy(position + tau_star*velocity)

    a = (1/2) * np.sum(velocity**2*precision)
    b = np.sum(position*precision*velocity) - np.sum(velocity*r)
    c = (1/2) * np.sum(position*precision*position) - np.sum(position*r) - u_star + np.log(V)

    delta = b**2 - 4*a*c
    tau_bound = (-b + np.sqrt(delta))/(2*a)
    return tau_bound

def simulate_arrival_time(position, velocity):
    tau_star = minimize_along_traj(position, velocity)
    tau = solve_for_time(position, velocity, tau_star)
    return tau

def run(position, lambda_ref=0.1, n_iter = 100000):
    h_positions = []
    h_velocities = []
    h_times = []
    h_positions.append(position[cpt])
    velocity = np.random.normal(size=len(position))
    h_velocities.append(velocity[cpt])
    refresh = 0
    for i in range(n_iter):
        if i % 1000 == 0:
            print("BPS")
            print(i)

        t_bounce = simulate_arrival_time(position, velocity)
        t_ref = np.random.exponential(lambda_ref)
        t = min(t_bounce, t_ref)
        position = position + t*velocity

        if t == t_ref:
            refresh += 1
            velocity = np.random.normal(size=len(position))
        else:
            velocity = bounce(position, velocity)

        h_positions.append(position[cpt])
        h_velocities.append(velocity[cpt])
        h_times.append(t)


    print("Refreshment rate:")
    print(refresh/n_iter)
    return np.array(h_positions), np.array(h_velocities), np.array(h_times)



def compute_bin(a,b, h_times, h_positions, h_velocities):
    h_velocities = h_velocities[:-1]
    h_positions = h_positions[:-1]

    a_bound = (a - h_positions)/h_velocities
    b_bound = (b - h_positions)/h_velocities

    lower = np.maximum(a_bound, np.zeros(len(h_velocities)))
    upper = np.minimum(b_bound, np.ones(len(h_velocities))*h_times)
    indx = (h_velocities > np.zeros(len(h_velocities))) * (a_bound < h_times) * (b_bound > np.zeros(len(h_velocities)))
    lower = lower[indx]
    upper = upper[indx]
    num_velo_pos = np.sum(upper - lower)

    upper = np.minimum(a_bound, np.ones(len(h_velocities))*h_times)
    lower = np.maximum(b_bound, np.zeros(len(h_velocities)))
    indx = (h_velocities < np.zeros(len(h_velocities))) * (a_bound > np.zeros(len(h_velocities))) *  (b_bound < h_times)
    lower = lower[indx]
    upper = upper[indx]
    num_velo_neg = np.sum(upper - lower)

    estimate = (num_velo_neg + num_velo_pos)/np.sum(h_times)
    return estimate


d = (lmax+1)**2 - 4


h_position, h_velocity, h_times = run(mu + 10*np.sqrt(sigma_real), n_iter=1000000, lambda_ref=0.5)


d = {"h_pos":h_position, "h_vel":h_velocity, "h_times":h_times}

np.save("test_bps.npy", d, allow_pickle=True)

exact_sampling = np.random.normal(loc=mu[cpt], scale=np.sqrt(sigma_real[cpt]), size=(1000000))


a = mu[cpt] + 0*np.sqrt(sigma_real[cpt])
b = mu[cpt] + 3*np.sqrt(sigma_real[cpt])
proba_BPS = compute_bin(a, b, h_times, h_position, h_velocity)
proba_exact = np.mean((exact_sampling > a) * (exact_sampling < b))

a_mu = -np.inf
b_mu = np.inf

mean_BPS = compute_bin(a_mu, b_mu, h_times, h_position, h_velocity)


print("Proba BPS", "[", a, b, "]")
print(proba_BPS)
print("Exact")
print(proba_exact)
print("Relative Error")
print(np.abs(proba_BPS-proba_exact)/proba_exact)

print("Mean exact")
print(np.mean(exact_sampling))
print("Mean BPS")
print(mean_BPS)
print(np.mean(h_position))
print("Real")
print(mu[cpt])

plt.plot(h_position, label="BPS", alpha = 0.5)
plt.plot(exact_sampling, label="Exact", alpha=0.5)
plt.legend(loc="upper right")
plt.show()


plot_acf(h_position, fft=True, title="BPS")
plot_acf(exact_sampling, fft=True, title="Exact")
plt.show()