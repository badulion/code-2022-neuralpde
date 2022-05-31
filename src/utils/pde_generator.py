from functools import partial
from typing import Optional
import numpy as np

from scipy.signal import convolve2d
from scipy.integrate import solve_ivp

import xarray as xr

from tqdm import tqdm

import os
import sys
import shutil
sys.path.append(".")


from src import utils
from src.utils.var import dict_hash
from rich.progress import track, Progress

class Filters:
    dx = [
        [0,0,0],
        [1,0,-1],
        [0,0,0],
    ]

    dy = [
        [0,-1,0],
        [0,0,0],
        [0,1,0],
    ]

    dx2 = [
        [0,0,0],
        [1,-2,1],
        [0,0,0],
    ]

    dy2 = [
        [0,1,0],
        [0,-2,0],
        [0,1,0],
    ]
    dxdy = [
        [-1,0,1],
        [0,0,0],
        [1,0,-1],
    ]

class PDEGenerator:
    def __init__(self, 
                 x_high, 
                 y_high, 
                 x_low, 
                 y_low, 
                 equation,
                 zero_levels,
                 params, 
                 variables,
                 tmin=0, 
                 tmax=10, 
                 tstep=1, 
                 seed=None,
                 verbose=False) -> None:
        # supplied variables
        self.x = x_high
        self.y = y_high
        self.x_low = x_low
        self.y_low = y_low
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.equation = equation

        # printing progress
        self.verbose = verbose

        # derived variables
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.size_x = len(self.x)
        self.size_y = len(self.y)

        self.tpoints = np.arange(tmin, tmax, self.tstep)
        if self.tmax % self.tstep == 0:
            self.tpoints = np.append(self.tpoints, tmax)

        # dynamic
        self.zero_levels = zero_levels
        self.params = params
        self.variables = variables

        # reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def _create_dynamic(self, equation, params, verbose=False):
        def dynamics_progress_wrapper(t, u, pbar, task, state, dynamics, params):
            last_t, = state
            pbar.update(task, advance=t-last_t)
            state[0] = t
            
            return dynamics(t,u,params)
        
        # selecting the derivative function
        if equation == "gas_dynamics":
            derivative = self._gas_dynamics_equation
        elif equation == "wave":
            derivative = self._wave_equation
        elif equation == "burgers":
            derivative = self._burgers_equation
        else:
            derivative = self._advection_diffusion_equation

        if verbose:
            out_dyn = partial(dynamics_progress_wrapper, dynamics=derivative, params=params)
        else: 
            out_dyn = partial(derivative, params=params)
        
        return out_dyn

    def generate(self, complexity=1, pbar=None, task=None):
        I = self._sum_of_gaussians(complexity=complexity, zero_levels=self.zero_levels)
        if self.equation == "wave": #second order equation
            I = np.concatenate([I, np.zeros_like(I)], axis=0)
        verbose = pbar is not None and task is not None
        self.dynamic = self._create_dynamic(self.equation, self.params, verbose=verbose)

        if verbose:
            sol = solve_ivp(
                self.dynamic, 
                [self.tmin,self.tmax], 
                np.reshape(I, (-1,)), 
                t_eval=self.tpoints, 
                method="RK23",
                args=[pbar, task, [self.tmin]]
            ).y
        else:
            sol = solve_ivp(
                self.dynamic, 
                [self.tmin,self.tmax], 
                np.reshape(I, (-1,)), 
                t_eval=self.tpoints, 
                method="RK23"
            ).y

        # fixing shapes (40000, *) -> (4, 100, 100, *)
        sol = np.reshape(sol, (-1, self.size_y, self.size_y, len(self.tpoints)))
    
        # fixing shapes (4, 100, 100, *) -> (4, *, 100, 100)
        sol = np.moveaxis(sol, -1, 1)

        # creating xarray
        num_variables = len(self.variables)
        dims = ("time", "x", "y")
        coords_dict = {
            'x': self.x, 
            'y': self.y, 
            'time': self.tpoints
        }
        x_dataarrays = {
            self.variables[i]: xr.DataArray(sol[i], dims=dims, coords=coords_dict) for i in range(num_variables)
        }

        x_dataset = xr.Dataset(
            x_dataarrays,
            attrs={
                "equation": self.equation,
            }
        )

        x_dataset = x_dataset.reindex(x=self.x_low, y=self.y_low, method='nearest')

        return x_dataset
        

    def _sum_of_gaussians(self, complexity=1, zero_levels = [0.0]):
        layers = []
        for zero_level in zero_levels:
            x_grid, y_grid = np.meshgrid(self.x, self.y)

            mx = [self.rng.uniform(np.min(self.x), np.max(self.x)) for i in range(complexity)]
            my = [self.rng.uniform(np.min(self.y), np.max(self.y)) for i in range(complexity)]

            gaussian = np.exp(-(20*(x_grid-np.mean(self.x))**2 + 20*(y_grid-np.mean(self.y))**2))

            u = zero_level+np.zeros_like(x_grid)
            for i in range(complexity):
                shift_x = np.argmin(np.abs(self.x-mx[i]))-np.argmin(np.abs(self.x-np.mean(self.x)))
                shift_y = np.argmin(np.abs(self.y-my[i]))-np.argmin(np.abs(self.y-np.mean(self.y)))
                
                component = np.roll(gaussian, (shift_y,shift_x), axis=(0,1))

                u = u + self.rng.uniform(-1, 1) * component
                
            #smoothing filter
            a = 5
            smoothing_filter = np.ones((2*a+1, 2*a+1))
            smoothing_filter/= np.sum(smoothing_filter)
            u = convolve2d(u, smoothing_filter, mode='same', boundary='wrap')
            layers.append(u)

        return np.stack(layers)


    def _gas_dynamics_equation(self, t, u, params):

        u = np.reshape(u, (-1, self.size_y, self.size_x))
        p = u[0,:,:]
        T = u[1,:,:]
        v = u[2,:,:]
        w = u[3,:,:]
        P = p * T/params['M']

        # partial derivatives
        dpdx = convolve2d(p, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        dvdx = convolve2d(v, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        dwdx = convolve2d(w, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        d2vdx2 = convolve2d(v, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        d2wdx2 = convolve2d(w, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        dTdx = convolve2d(T, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        d2Tdx2 = convolve2d(T, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        dPdx = convolve2d(P, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)

        dpdy = convolve2d(p, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)
        dvdy = convolve2d(v, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)
        dwdy = convolve2d(w, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)
        d2vdy2 = convolve2d(v, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)
        d2wdy2 = convolve2d(w, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)
        dTdy = convolve2d(T, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)
        d2Tdy2 = convolve2d(T, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)
        dPdy = convolve2d(P, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)

        d2vdxdy = convolve2d(v, Filters.dxdy, mode='same', boundary='wrap')/(4*self.dy*self.dx)
        d2wdxdy = convolve2d(w, Filters.dxdy, mode='same', boundary='wrap')/(4*self.dy*self.dx)

        #resulting dynamic
        dpdt = - v*dpdx - w*dpdy - p*(dvdx+dwdy)
        dTdt = - v*dTdx - w*dTdy - params['gamma']*T*(dvdx+dwdy) +params['gamma']*params['k'] / p *(d2Tdx2+d2Tdy2)
        
        dvdt = - v*dvdx - w*dvdy - (dPdx / p) + (params['mu'] / p)*(d2vdx2+d2wdxdy)
        dwdt = - v*dwdx - w*dwdy - (dPdy / p) + (params['mu'] / p)*(d2wdy2+d2vdxdy)

        dudt = params['scale']*np.stack([dpdt, dTdt, dvdt, dwdt])
        return np.reshape(dudt, (-1,))

    def _wave_equation(self, t, u, params):       
        u = np.reshape(u, (-1, self.size_y, self.size_x))
        v = u[0,:,:]
        v1 = u[1,:,:]

        # partial derivatives
        d2udx2 = convolve2d(v, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        d2udy2 = convolve2d(v, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)

        #resulting dynamic
        d2vudt2 = params['scale']*(params['c']**2) * (d2udx2 + d2udy2)
        dudt = np.concatenate([v1, d2vudt2])
        return np.reshape(dudt, (-1,))

    def _advection_diffusion_equation(self, t, u, params):
        u = np.reshape(u, (self.size_y, self.size_x))

        # partial derivatives
        dudx = convolve2d(u, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        dudy = convolve2d(u, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)

        d2udx2 = convolve2d(u, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        d2udy2 = convolve2d(u, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)

        dudt = params['scale']*(params['D']*(d2udx2+d2udy2)-(params['vx']*dudx + params['vy']*dudy))
        return np.reshape(dudt, (-1,))

    def _burgers_equation(self, t, u, params):
        u = np.reshape(u, (-1, self.size_y, self.size_x))
        v = u[0,:,:]
        w = u[1,:,:]

        # partial derivatives
        dvdx = convolve2d(v, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        dvdy = convolve2d(v, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)
        dwdx = convolve2d(w, Filters.dx, mode='same', boundary='wrap')/(2*self.dx)
        dwdy = convolve2d(w, Filters.dy, mode='same', boundary='wrap')/(2*self.dy)

        d2vdx2 = convolve2d(v, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        d2vdy2 = convolve2d(v, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)
        d2wdx2 = convolve2d(w, Filters.dx2, mode='same', boundary='wrap')/(self.dx**2)
        d2wdy2 = convolve2d(w, Filters.dy2, mode='same', boundary='wrap')/(self.dy**2)

        # resulting dynamic
        dvdt = - v * dvdx - w * dvdy + params['c']*(d2vdx2+d2vdy2)
        dwdt = - v * dwdx - w * dwdy + params['c']*(d2wdx2+d2wdy2)

        dudt = params['scale']*np.stack([dvdt, dwdt])

        return np.reshape(dudt, (-1,))

    def _trigonometric_init(self, x, y, components=5, zero=0.1):
        sigma = [np.random.uniform(0.1, 0.45) for _ in range(components)]
        mx = [np.random.uniform(sigma[i], 1-sigma[i]) for i in range(components)]
        my = [np.random.uniform(sigma[i], 1-sigma[i]) for i in range(components)]
        signs = [1 if np.random.uniform(0, 1) < 0.5 else -1 for _ in range(components)]
        scale = [np.random.uniform(0.1, 0.5) for _ in range(components)]

        curves = []
        for i in range(components):
            d = np.sqrt((x-mx[i])**2+(y-my[i])**2)
            curve = scale[i]*(1+np.cos(d*np.pi/sigma[i]))*(d<sigma[i])
            curves.append(signs[i]*curve)
        return sum(curves)+zero

class GenerationUtility:
    def __init__(self,
                 data_dir,
                 name,
                 zero_levels,
                 variables,
                 params,
                 complexities,
                 num_equations,
                 domain,
                 verbose=False) -> None:

        # parameters
        self.name = name
        self.zero_levels = list(zero_levels)
        self.variables = list(variables)
        self.params = params
        self.complexities = complexities
        self.num_equations = num_equations
        self.domain = domain
        self.verbose = verbose

        # domain
        self.x_high = np.linspace(domain.x.min, domain.x.max, domain.x.points, endpoint=False)
        self.y_high = np.linspace(domain.y.min, domain.y.max, domain.y.points, endpoint=False)

        self.x_low = np.linspace(domain.x.min, domain.x.max, domain.x.points_low, endpoint=False)
        self.y_low = np.linspace(domain.y.min, domain.y.max, domain.y.points_low, endpoint=False)

        self.tmin=domain.t.min
        self.tmax=domain.t.max
        self.tstep=domain.t.step

        # output path
        self.data_dir = data_dir


        # instantiating the generator
        self.generator = PDEGenerator(
            x_high=self.x_high,
            y_high=self.y_high,
            x_low=self.x_low,
            y_low=self.y_low,
            equation=self.name,
            zero_levels=zero_levels,
            params=params,
            variables = variables,
            tmin=self.tmin,
            tmax=self.tmax,
            tstep=self.tstep,
        )

    def generate(self):
        log = utils.get_logger(__name__)
        if os.path.exists(self.data_dir):
            log.info("Deleting previous equations.")
            shutil.rmtree(self.data_dir)

        # generate
        log.info("Solving train equations.")
        complexity = self.complexities.train
        eq_hash = dict_hash(
            {'name': self.name,
             'complexity': complexity,
             'zero_levels': self.zero_levels,
             'variables': self.variables},
            self.params,
        )
        base_path = os.path.join(self.data_dir, "train")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with Progress() as pbar:
            all_train_equations_task = pbar.add_task("[red]Solving...", total=self.num_equations.train)
            if self.verbose:
                single_train_equation_task = pbar.add_task("[purple]Equation progress...", total=self.tmax-self.tmin)
            for i in range(self.num_equations.train):
                if self.verbose:
                    eq_dataset = self.generator.generate(complexity=complexity, pbar=pbar, task=single_train_equation_task)
                    pbar.reset(single_train_equation_task)
                else: 
                    eq_dataset = self.generator.generate(complexity=complexity)
                path = os.path.join(base_path, f"{eq_hash}_{i}.nc")
                eq_dataset.to_netcdf(path)
                pbar.advance(all_train_equations_task, advance=1)

        
        log.info("Solving validation equations.")
        complexity = self.complexities.val
        eq_hash = dict_hash(
            {'name': self.name,
             'complexity': complexity,
             'zero_levels': self.zero_levels,
             'variables': self.variables},
            self.params,
        )
        base_path = os.path.join(self.data_dir, "val")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with Progress() as pbar:
            all_train_equations_task = pbar.add_task("[blue]Solving...", total=self.num_equations.val)
            if self.verbose:
                single_train_equation_task = pbar.add_task("[purple]Equation progress...", total=self.tmax-self.tmin)
            for i in range(self.num_equations.val):
                if self.verbose:
                    eq_dataset = self.generator.generate(complexity=complexity, pbar=pbar, task=single_train_equation_task)
                    pbar.reset(single_train_equation_task)
                else: 
                    eq_dataset = self.generator.generate(complexity=complexity)
                path = os.path.join(base_path, f"{eq_hash}_{i}.nc")
                eq_dataset.to_netcdf(path)
                pbar.advance(all_train_equations_task, advance=1)
        
        log.info("Solving test equations.")
        complexity = self.complexities.test
        eq_hash = dict_hash(
            {'name': self.name,
             'complexity': complexity,
             'zero_levels': self.zero_levels,
             'variables': self.variables},
            self.params,
        )
        base_path = os.path.join(self.data_dir, "test")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with Progress() as pbar:
            all_train_equations_task = pbar.add_task("[green]Solving...", total=self.num_equations.test)
            if self.verbose:
                single_train_equation_task = pbar.add_task("[purple]Equation progress...", total=self.tmax-self.tmin)
            for i in range(self.num_equations.test):
                if self.verbose:
                    eq_dataset = self.generator.generate(complexity=complexity, pbar=pbar, task=single_train_equation_task)
                    pbar.reset(single_train_equation_task)
                else: 
                    eq_dataset = self.generator.generate(complexity=complexity)
                path = os.path.join(base_path, f"{eq_hash}_{i}.nc")
                eq_dataset.to_netcdf(path)
                pbar.advance(all_train_equations_task, advance=1)


if __name__ == "__main__":

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)

    gen = PDEGenerator(
                 x_high=x, 
                 y_high=y, 
                 x_low=x, 
                 y_low=y, 
                 equation="gas_dynamics",
                 zero_levels=[2,2,0,0],
                 params={
                    "scale": 0.002,
                    "mu": 0.01,
                    "k": 0.01,
                    "gamma": 1,
                    "M": 1,
                 }, 
                 variables=["density", "temperature", "velocity_x", "velocity_y"],
                 tmin=0, 
                 tmax=100, 
                 tstep=1, 
                 seed=None)
    params = {
        'scale': 0.002,
        'mu': 0.01,
        'k': 0.01,
        'gamma': 1,
        'M': 1
    }

    xr_dataset = gen.generate(complexity=5)
    quit()
    xr_dataset.to_netcdf("example.nc")
