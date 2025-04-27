import numpy as np

class GenerateData:
    def __init__(self, m):
        """
        M: number of time series to generate -> [int]
        """

        self.m = m

    def generate_ar(self, s1=0.1, s2=0.05, s3=0.05, b=0.7, b1=-1, b2=-1, x0=0):
        """
        s1, s2, s3: variances of the noises, must be > 0 -> [float]
        b, b1, b2: model parameters : -> [float]
        """
        eps1 = np.random.normal(0, s1, self.m)
        eps2 = np.random.normal(0, s2, self.m)
        eps3 = np.random.normal(0, s3, self.m)

        xt1 = b + eps1
        xt2 = b1 * xt1 + eps2
        xt3 = b2 * xt2 + np.sqrt(np.abs(xt1)) + eps3

        x = np.column_stack((xt1, xt2, xt3))
        x_ar = np.zeros((self.m, 4))
        x_ar[:, 0], x_ar[:, 1:] = x0, x
        return x_ar

    def generate_garch(self, n, alpha_0=5, alpha_1=0.4, alpha_2=0.1, s=0.1, x0=0):
        """
        n: length of time series to generate -> [int] : default = 60
        s: variance of the noise -> [float] : default = 0.1
        alpha0, alpha1, alpha2: model parameters : -> [float]
        """

        def simulate():
            time_series = list()
            x_next = 0.0
            x_prev = 0.0

            for t in range(n + 50):
                if t >= 50:
                    time_series.append(x_next)

                sigma = np.sqrt(alpha_0 + alpha_1 * x_next ** 2 + alpha_2 * x_prev ** 2)
                x_prev = x_next
                x_next = sigma * np.random.normal(scale=s)

            return time_series

        x = np.array([simulate() for i in range(self.m)])
        x_garch = np.zeros((self.m, n + 1))
        x_garch[:, 0], x_garch[:, 1:] = x0, x
        return x_garch

    def generate_ou(self, theta_range, mu_range, sigma_range, n, dt=1 / 252, x0=1):
        """
        theta_range: range of theta values -> [list of two ints]
        sigma_range: range of sigma values -> [list of two ints]
        mu_range: range of mu values -> [list of two ints]
        n: time series length -> [int]
        dt: time step -> [float] : default = 0.01
        X0: initial value -> [float] : default = 1
        """

        def simulate(theta, mu, sigma):
            x = np.zeros(n + 1)
            x[0] = x0
            for t in range(1, n + 1):
                mu_t = x[t - 1] * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
                sigma_t = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
                x[t] = mu_t + np.sqrt(sigma_t) * np.random.normal(0, 1)
            return x

        thetas = np.random.uniform(theta_range[0], theta_range[1], self.m)
        mus = np.random.uniform(mu_range[0], mu_range[1], self.m)
        sigmas = np.random.uniform(sigma_range[0], sigma_range[1], self.m)
        return np.array([simulate(thetas[i], mus[i], sigmas[i]) for i in range(self.m)])

    def generate_bs(self, r_range, sigma_range, n, dt=1 / 252, x0=1):
        """
        r_range: range of r values -> [list of two ints]
        sigma_range: range of sigma values -> [list of two ints]
        n: time series length -> [int]
        dt: time step Euler
        X0: initial value -> [float] : default = 0
        """
        z = np.random.normal(0, 1, (self.m, n))
        x = np.zeros((self.m, n + 1))
        x[:, 0] = x0
        r = np.random.uniform(r_range[0], r_range[1], self.m)
        sigma = np.random.uniform(sigma_range[0], sigma_range[1], self.m)

        for i in range(n):
            x[:, i + 1] = x[:, i] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * z[:, i])
        return x

    def generate_heston_vol(self, kappa_range, theta_range, xi_range, n, dt=1 / 252, v0=1):
        vol_heston = np.zeros((self.m, n + 1))

        def simulate(kappa, theta, xi, dt=dt):
            vol = np.zeros(n + 1)
            v_t = v0
            vol[0] = v0
            for t in range(1, n + 1):
                wt = np.random.randn(1) * np.sqrt(dt)
                v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * wt)
                vol[t] = v_t

            return vol

        kappa = np.random.uniform(kappa_range[0], kappa_range[1], self.m)
        theta = np.random.uniform(theta_range[0], theta_range[1], self.m)
        xi = np.random.uniform(xi_range[0], xi_range[1], self.m)

        for i in range(self.m):
            vol = simulate(kappa[i], theta[i], xi[i])
            vol_heston[i] = vol

        return vol_heston

    def generate_heston(self, r_range, kappa_range, theta_range, rho_range, xi_range, n, dt=1/252, s0=1, v0=1):
        """
        r/k/theta/pho/xi_range: range of params values -> [list of two ints]
        N: time series length -> [int]
        T: terminal time -> [int] : default = 1
        S0: price initial value -> [float] : default = 1
        v0: price initial value -> [float] : default = 0
        """

        def simulate_ig(mu, lambda_):
            g = np.random.randn()
            y = g ** 2
            x = mu + (0.5 / lambda_) * (mu ** 2 * y - mu * np.sqrt(4 * mu * lambda_ * y + (mu * y) ** 2))
            u = np.random.uniform()

            if u <= mu / (mu + x):
                return x
            return mu ** 2 / x

        def simulate_vol(kappa, theta, xi):
            v, u, z = np.zeros(n+1), np.zeros(n), np.zeros(n)
            v[0] = v0
            a, b = kappa * theta, -kappa
            for t in range(n):
                alpha_t = v[t] * (np.exp(b * dt) - 1) / b + a * ((np.exp(b * dt) - 1) / b - dt) / b
                sigma_t = xi / b * (np.exp(b * dt) - 1)
                u[t] = simulate_ig(alpha_t, (alpha_t / sigma_t) ** 2)
                z[t] = 1 / sigma_t * (u[t] - alpha_t)
                v[t + 1] = v[t] + a * dt + b * u[t] + xi * z[t]

            return v, u, z

        def simulate_h(r, kappa, theta, rho, xi):
            v, u, z = simulate_vol(kappa, theta, xi)
            log_s = np.zeros(n+1)
            log_s[0] = np.log(s0)
            sq_pho = np.sqrt(1 - rho ** 2)
            for t in range(n):
                log_s[t+1] = log_s[t] + r * dt - 0.5 * u[t] + rho * z[t] + sq_pho * np.sqrt(u[t]) * np.random.randn()
            return np.exp(log_s), v

        heston = np.zeros((self.m, n + 1, 2))

        r = np.random.uniform(r_range[0], r_range[1], self.m)
        kappa = np.random.uniform(kappa_range[0], kappa_range[1], self.m)
        theta = np.random.uniform(theta_range[0], theta_range[1], self.m)
        rho = np.random.uniform(rho_range[0], rho_range[1], self.m)
        xi = np.random.uniform(xi_range[0], xi_range[1], self.m)

        for i in range(self.m):
            price, vol = simulate_h(r[i], kappa[i], theta[i], rho[i], xi[i])
            serie = np.concatenate([price[:, np.newaxis], vol[:, np.newaxis]], axis=1)
            heston[i] = serie

        return heston

    def generate_heston_2(self, r_range, kappa_range, theta_range, rho_range, xi_range, n, dt=1/252, s0=1, v0=1):
        """
        r/k/theta/pho/xi_range: range of params values -> [list of two ints]
        N: time series length -> [int]
        T: terminal time -> [int] : default = 1
        S0: price initial value -> [float] : default = 1
        v0: price initial value -> [float] : default = 0
        """
        heston = np.zeros((self.m, n + 1, 2))

        def simulate(r, kappa, theta, rho, xi, dt=dt):
            prices, vola = np.zeros(n + 1), np.zeros(n + 1)
            s_t, v_t = s0, v0
            prices[0], vola[0] = s0, v0
            for t in range(1, n + 1):
                wt = np.random.multivariate_normal(np.array([0, 0]),
                                                   cov=np.array([[1, rho],
                                                                 [rho, 1]])) * np.sqrt(dt)

                s_t = s_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * wt[0]))
                v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * wt[1])
                prices[t] = s_t
                vol[t] = v_t

            return prices[:, np.newaxis], vola[:, np.newaxis]

        r = np.random.uniform(r_range[0], r_range[1], self.m)
        kappa = np.random.uniform(kappa_range[0], kappa_range[1], self.m)
        theta = np.random.uniform(theta_range[0], theta_range[1], self.m)
        rho = np.random.uniform(rho_range[0], rho_range[1], self.m)
        xi = np.random.uniform(xi_range[0], xi_range[1], self.m)

        for i in range(self.m):
            price, vol = simulate(r[i], kappa[i], theta[i], rho[i], xi[i])
            serie = np.concatenate([price, vol], axis=1)
            heston[i] = serie

        return heston

    def generate_sine(self, n, d, x0=0):

        data = np.zeros((self.m, n + 1, d))
        data[:, 0] = x0

        for i in range(self.m):
            for k in range(d):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                temp_data = [np.sin(freq * j + phase) for j in range(n)]
                data[i, 1:, k] = temp_data
            data[i] = (data[i] + 1) * 0.5

        return data

    def generate_ar_multi(self, n, d, phi, sigma, x0=0):
        """
        Generate sequences from an autoregressive multivariate Gaussian model.
        Parameters:
            phi (float): Autoregressive coefficient in [0, 1].
            sigma (float): Controls the correlation across features, in [-1, 1].
            d (int): Number of features.
            n (int): Number of time steps to generate.
            x0: initial value
        """

        data = np.zeros((self.m, n + 1, d))
        data[:, 0] = x0

        sigma_matrix = sigma * np.ones((d, d)) + (1 - sigma) * np.eye(d)
        noise = np.random.multivariate_normal(mean=np.zeros(d), cov=sigma_matrix, size=(self.m, n))
        for t in range(1, n + 1):
            data[:, t] = phi * data[:, t - 1] + noise[:, t - 1]

        return data