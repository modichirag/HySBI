import numpy as np
from scipy.special import p_roots
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps, romberg
#
from nbodykit.cosmology import LinearPower, Planck15



def sigmasqr(R, pk, kmin=0.00001, kmax=1000.0):
    """Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc
    .. math::
       \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)
    where
    .. math::
       W(kR) = \\frac{3j_1(kR)}{kR}
    Code taken from jax-cosmo: https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/master/jax_cosmo/power.py
    """

    def int_sigma(logk):
        k = np.exp(logk)
        x = k * R
        w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x * x * x)
        pk_arr = pk(k) 
        return k * (k * w) ** 2 * pk_arr

    y = romberg(int_sigma, np.log10(kmin), np.log10(kmax), divmax=10)
    return 1.0 / (2.0 * np.pi ** 2.0) * y


class NBPklin:
    """
    Wrapper for generating linear power spectrum from Nbodykit    
    """
    def __init__(self, cosmo=None, transfer='CLASS', z=0.):
        self.k_arr = np.geomspace(1e-6,20,1000)
        self.z = z 
        self.transfer = transfer
        if cosmo is None:
            self.cosmo = Planck15
        else:
            self.cosmo = cosmo
        self.setup_defaults()

    def setup_defaults(self):
        pklin = LinearPower(self.cosmo, self.z, transfer=self.transfer)
        self.cosmo.sigma_8 = sigmasqr(8, pklin)**0.5
        self.cosmo_params = ['$Omega_m$', '$Omega_b$''$h$', '$n_s$', '$sigma_8$']
        self.default_cosmo = tuple([self.cosmo.Omega0_m, self.cosmo.Omega0_b, self.cosmo.h, self.cosmo.n_s, self.cosmo.sigma_8])


    def __call__(self, Omega0_m=None, Omega0_b=None, h=None, n_s=None, sigma_8=None):
        """
        Return interpolation function for linear power spectrum at sepcified cosmology. 
        Unspecified parameters are set to default parameters of the class instance.    
        """
        if Omega0_m is None: Omega0_m = self.default_cosmo[0]
        if Omega0_b is None: Omega0_b = self.default_cosmo[1]
        if h is None: h = self.default_cosmo[2]
        if n_s is None: n_s = self.default_cosmo[3]
        if sigma_8 is None: sigma_8 = self.default_cosmo[4]
        
        Omega0_cdm = Omega0_m - Omega0_b
        cosmo = self.cosmo.clone(Omega0_cdm=Omega0_cdm, Omega0_b=Omega0_b, h=h, n_s=n_s)
        pklin = LinearPower(cosmo, self.z, transfer=self.transfer)
        amplitude = sigmasqr(8, pklin)**0.5
        pk_arr = pklin(self.k_arr) *sigma_8**2 / amplitude**2
        pkl = InterpolatedUnivariateSpline(self.k_arr, pk_arr, ext='zeros')
        return pkl


class PkMatter:
    
    def __init__(self):
        """
        Class for estimating matter power spectrum, refactored from functions provided by Oliver 
        """

        self.setup()
        

    def setup(self):
        # setup arrays to be used
        self.k_arr = np.geomspace(1e-6,10,100)
        self.r_arr = np.geomspace(1e-5,9e4,1001)[None,:]
        mu_arr, w_arr = p_roots(100) # Gauss-Legendre quadrature weights
        self.mu_arr = mu_arr[None,None,:]
        self.w_arr = w_arr[None,None,:]        
        self.psi_arr = np.sqrt(1.+self.r_arr[:,:,None]**2-2*self.r_arr[:,:,None]*self.mu_arr)
        
        # setup factors for P22
        self.F_2d = lambda r, mu: (7.*mu+(3.-10.*mu**2)*r)/(14.*r*(r**2-2.*r*mu+1.))
        self.F_2d_r_mu_factor = np.abs(self.F_2d(self.r_arr[:,:,None],self.mu_arr))**2.
        self.P22_pref = self.k_arr**3./(2.*np.pi**2)
        
        # setup factors for P13
        self.P13_integ = lambda r: 12./r**4.-158./r**2.+100.-42.*r**2+3./r**5.*(7.*r**2+2)*(r**2-1.)**3.*np.log(np.abs((r+1.)/(r-1.)))        
        self.P13_integ_rarr = self.P13_integ(self.r_arr)


    def p13(self, pkl):

        P13_pref = self.k_arr**3/(252.*(2.*np.pi)**2)*pkl(self.k_arr)
        integrand = self.r_arr**2*pkl(self.r_arr*self.k_arr[:,None])*self.P13_integ_rarr
        P13_raw = P13_pref*simps(integrand, self.r_arr)
        P13 = InterpolatedUnivariateSpline(self.k_arr, P13_raw, ext='zeros')
        return P13
            
    def p22(self, pkl):
        
        mu_integral = np.sum(self.w_arr* pkl(self.k_arr[:,None,None]*self.psi_arr) *self.F_2d_r_mu_factor, axis=2)
        P22_raw = self.P22_pref * simps(self.r_arr**2* pkl(self.r_arr*self.k_arr[:,None])* mu_integral, self.r_arr, axis=1)
        P22 = InterpolatedUnivariateSpline(self.k_arr, P22_raw, ext='zeros')
        return P22
    
    def pct(self, pkl):
        
        return lambda k,cs: -2.*cs*k**2*pkl(k)

        


