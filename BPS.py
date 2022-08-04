import config
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class BPS():
    def __init__(self, nside, lmax, noise, beam, obs_map, lambda_ref, n_iter=1, within_gibbs=True):
        self.nside = nside
        self.lmax = lmax
        #self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        ###Careful !!!
        self.noise = np.array([noise for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.beam = beam
        self.pix_map = obs_map
        self.n_iter = n_iter
        self.within_gibbs = within_gibbs
        self.lambda_ref = lambda_ref
        self.equals_zeros = 0


    def compute_energy(self, position, var_cls):
        precision = self.beam**2/self.noise + 1/var_cls
        r = self.beam*(1/self.noise)*self.pix_map
        return (1/2)*np.sum(position*precision*position) - np.sum(position*r)

    def compute_gradient_energy(self, alm_map, var_cls):
        precision = (self.beam**2/self.noise + 1/var_cls)
        r = self.beam*(1/self.noise)*self.pix_map
        return precision*alm_map - r

    def bounce(self, position, velocity, var_cls):
        grad = self.compute_gradient_energy(position, var_cls)
        dot_prod = np.sum(grad*velocity)
        norm_squared_grad = np.sum(grad**2)
        new_velocity = velocity - 2*(dot_prod/norm_squared_grad)*grad
        return new_velocity

    def minimise_along_traj(self, position, velocity, var_cls):
        precision = 1/var_cls + self.beam**2/self.noise
        r = self.beam*(1/self.noise)*self.pix_map
        num = np.sum(velocity*r) - np.sum(position*precision*velocity)
        denom = np.sum(velocity*precision*velocity)
        tau_star = max(num/denom, 0)

        U_star = self.compute_energy(position + tau_star * velocity, var_cls)
        low_bound = tau_star - 0.1
        up_bound = tau_star + 0.1

        if False:
            x = np.linspace(low_bound, up_bound, num=10000)
            h_y = []
            for t in x:
                y = self.compute_energy(position + t * velocity, var_cls)
                h_y.append(y)

            plt.plot(x, h_y)
            plt.axvline(x=tau_star, color="blue")
            plt.axhline(y=U_star)
            plt.show()

        return tau_star

    def solve_for_time(self, position, velocity, var_cls, tau_star):

        precision = 1/var_cls + self.beam**2/self.noise
        r = self.beam*(1/self.noise)*self.pix_map
        V = np.random.uniform()

        a = (1 / 2) * np.sum(velocity * precision * velocity)
        b = np.sum(position * precision * velocity) - np.sum(velocity * r)
        if tau_star == 0:
            self.equals_zeros +=1
            c = np.log(V)

        else:
            #c = np.log(V) + a*tau_star**2 + tau_star*b
            c = np.log(V) - a * tau_star ** 2 - tau_star * b

        delta = b ** 2 - 4 * a * c
        tau = (- b + np.sqrt(delta)) / (2 * a)
        if delta < 0:
            U_star = self.compute_energy(position + tau_star * velocity, var_cls)
            low_bound = tau_star - 0.1
            up_bound = tau_star + 0.1

            x = np.linspace(low_bound, up_bound, num = 10000)
            h_y = []
            for t in x:
                y = self.compute_energy(position + t*velocity, var_cls)
                h_y.append(y - U_star)


            plt.plot(x, h_y)
            plt.axvline(x=tau_star, color="blue")
            plt.axhline(y=-np.log(V))
            plt.axvline(x=tau, color="red")
            plt.show()

        return tau

    def simulate_arrival_time(self, position, velocity, var_cls):
        tau_star = self.minimise_along_traj(position, velocity, var_cls)
        tau = self.solve_for_time(position, velocity, var_cls, tau_star)
        return tau

    """
    def run_step(self, position, velocity, var_cls):
        tau_bounce = self.simulate_arrival_time(position, velocity, var_cls)
        tau_ref = np.random.exponential(self.lambda_ref)
        tau = min(tau_bounce, tau_ref)
        position_new = position + tau*velocity
        refresh = 0
        if tau == tau_ref:
            new_velocity = np.random.normal(size=len(position_new))
            refresh = 1
        else:
            new_velocity = self.bounce(position, velocity, var_cls)

        return position_new, new_velocity, refresh, tau

    def run(self, position, var_cls):
        h_refresh = []
        h_velocities = []
        h_positions = []
        h_times = []
        velocity = np.random.normal(size=len(position))
        h_positions.append(position[config.cpt])
        h_velocities.append(velocity[config.cpt])
        for i in range(self.n_iter):
            if i % 100== 0:
                print("BPS")
                print(i)

            position, velocity, refresh, tau = self.run_step(position, velocity, var_cls)
            h_refresh.append(refresh)
            h_positions.append(position[config.cpt])
            h_velocities.append(velocity[config.cpt])
            h_times.append(tau)

        print("Refresh rate")
        print(np.mean(h_refresh))
        return np.array(h_positions), np.array(h_velocities), np.array(h_times)
    """

    def run(self, position, var_cls):
        h_positions = []
        h_velocities = []
        h_times = []
        h_positions.append(position[cpt])
        velocity = np.random.normal(size=len(position))
        h_velocities.append(velocity[config.cpt])
        refresh = 0
        for i in range(self.n_iter):
            if i % 1000 == 0:
                print("BPS")
                print(i)

            t_bounce = self.simulate_arrival_time(position, velocity, var_cls)
            t_ref = np.random.exponential(self.lambda_ref)
            t = min(t_bounce, t_ref)
            position = position + t * velocity

            if t == t_ref:
                refresh += 1
                velocity = np.random.normal(size=len(position))
            else:
                velocity = bounce(position, velocity)

            h_positions.append(position[cpt])
            h_velocities.append(velocity[cpt])
            h_times.append(t)

        print("Refreshment rate:")
        print(refresh / self.n_iter)
        return np.array(h_positions), np.array(h_velocities), np.array(h_times)




cpt = 0


def simulate_bounce_time(position, velocity):
    prod_scal = np.sum(position*velocity)
    norm_vel_square = np.sum(velocity**2)
    log_V = np.log(np.random.uniform())
    if prod_scal <= 0:
        return (-prod_scal + np.sqrt(-norm_vel_square*log_V))/norm_vel_square

    else:
        return (-prod_scal + np.sqrt(prod_scal**2 - norm_vel_square*log_V))/norm_vel_square


def compute_energy(position):
    return np.sum(position**2)

def compute_grad_energy(position):
    return 2*position

def bounce(position, velocity):
    return velocity - 2* (np.sum(position*velocity)/np.sum(position**2))*position


def run(position, lambda_ref, n_iter = 1000000):
    h_positions = []
    h_velocities = []
    h_times = []
    h_positions.append(position[cpt])
    velocity = np.random.normal(size=len(position))
    h_velocities.append(velocity[config.cpt])
    refresh = 0
    for i in range(n_iter):
        if i % 1000 == 0:
            print("BPS")
            print(i)

        t_bounce = simulate_bounce_time(position, velocity)
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


"""
nside = 512
Npix = 12*nside**2
lmax = 2*nside
noise = 40**2*(4*np.pi)/Npix
noise_covar = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])

import healpy as hp
import utils
fwhm_radians = (np.pi / 180) * 0.35
bl_gauss = hp.gauss_beam(fwhm_radians, lmax=lmax)[2:]
beam = np.array([bl for l, bl in enumerate(bl_gauss, start=2) for i in range(2 * l + 1)])

theta_true = config.COSMO_PARAMS_MEAN + config.COSMO_PARAMS_SIGMA
cls_true = utils.generate_cls(theta_true)[:lmax-1]
var_cl_full = utils.generate_var_cls(cls_true)
var_cl_full = np.ones(len(var_cl_full))*4

beam = np.ones(len(beam))
obs_map = np.zeros(len(beam))
lambda_ref = 0.0005
bounce_sim = BPS(config.NSIDE, config.L_MAX_SCALARS, 1/4, beam, obs_map, lambda_ref, n_iter=1000, within_gibbs=True)
h_pos, h_velocities, h_times = bounce_sim.run(np.zeros(len(beam)), var_cl_full)
"""

"""
d = 1000
lambda_ref = 0.5
position = np.ones(d)*0.5
h_pos, h_velocities, h_times = run(position, lambda_ref, 1000000)

burnin = 0
h_positions = h_pos
h_velocities2 = h_velocities[burnin:-1]
h_positions2 = h_positions[burnin:-1]
first_term = h_times[burnin:]*h_positions2**2
second_term = h_velocities2*h_positions2*h_times[burnin:]**2
third_term = (1/3)*h_velocities2**2*h_times[burnin:]**3
integrals = first_term + second_term + third_term
T = np.sum(h_times[burnin:])
estim_variances = (1/T)*np.sum(integrals)
print("Estimate variance")
print(estim_variances)

a = 1/2
b = 3
estimed_bin = compute_bin(a, b, h_times, h_pos, h_velocities)

print("Estimed proba")
print(estimed_bin)
print("True")
true = stats.norm.cdf(b, loc=0, scale=1/np.sqrt(2)) - stats.norm.cdf(a, loc=0, scale=1/np.sqrt(2))
print(true)
print("Rel error")
print(np.abs(true - estimed_bin)/true)


h_exact = np.random.normal(loc=0, scale = 1/np.sqrt(2), size = (1000000, 1))

plt.plot(h_pos, alpha=0.5, label="BPS")
plt.plot(h_exact, alpha=0.5, label="Exact")
plt.legend(loc="upper right")
plt.show()


plt.hist(h_pos, alpha=0.5, label="BPS", bins = 50, density=True)
plt.hist(h_exact, alpha=0.5, label="Exact", bins = 50, density=True)
plt.legend(loc="upper right")
plt.show()
"""
