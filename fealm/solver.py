#
# Multiprocessing version of ParticleSwarm based on Pymanopt's ParticleSwarm
#

import time

import numpy as np
import numpy.random as rnd

from pymanopt.solvers.solver import Solver

# faster but memory leak can happen
from pathos.multiprocessing import ProcessPool as Pool
# slower but memory leak safe
# from pathos.multiprocessing import _ProcessingPool as _Pool


class ParticleSwarm(Solver):
    """Particle swarm optimization (PSO) method.
    Perform optimization using the derivative-free particle swarm optimization
    algorithm.
    Args:
        max_cost_evaluations: Maximum number of allowed cost evaluations.
        max_iterations: Maximum number of allowed iterations.
        population_size: Size of the considered swarm population.
        nostalgia: Quantifies performance relative to past performances.
        social: Quantifies performance relative to neighbors.
    """

    def __init__(
        self,
        max_cost_evaluations=None,
        max_iterations=None,
        population_size=None,
        nostalgia=1.4,
        social=1.4,
        n_jobs=-1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._max_cost_evaluations = max_cost_evaluations
        self._max_iterations = max_iterations
        self._population_size = population_size
        self._nostalgia = nostalgia
        self._social = social
        self._n_jobs = n_jobs

    def solve(self, problem, x=None, multiple_answers=False):
        """Run PSO algorithm.
        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
            x: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        man = problem.manifold
        verbosity = self._verbosity
        objective = problem.cost

        # Choose proper default algorithm parameters. We need to know about the
        # dimension of the manifold to limit the parameter range, so we have to
        # defer proper initialization until this point.
        dim = man.dim
        if self._max_cost_evaluations is None:
            self._max_cost_evaluations = max(5000, 2 * dim)
        if self._max_iterations is None:
            self._max_iterations = max(500, 4 * dim)
        if self._population_size is None:
            self._population_size = min(40, 10 * dim)

        # If no initial population x is given by the user, generate one at
        # random.
        if x is None:
            x = [man.rand() for i in range(int(self._population_size))]
        elif not hasattr(x, "__iter__"):
            raise ValueError("The initial population x must be iterable")
        else:
            if len(x) != self._population_size:
                print("The population size was forced to the size of "
                      "the given initial population")
                self._population_size = len(x)

        # Initialize personal best positions to the initial population.
        y = list(x)

        # Save a copy of the swarm at the previous iteration.
        xprev = list(x)

        if self._n_jobs > 0:
            with Pool(nodes=self._n_jobs) as p:
                # Initialize velocities for each particle.
                v = [man.randvec(xi) for xi in x]
                # Compute cost for each particle xi.
                costs = p.map(objective, x)
        else:
            with Pool() as p:
                # Initialize velocities for each particle.
                v = [man.randvec(xi) for xi in x]
                # Compute cost for each particle xi.
                costs = p.map(objective, x)

        fy = list(costs)
        cost_evaluations = self._population_size

        # Identify the best particle and store its cost/position.
        imin = np.array(costs).argmin()
        fbest = costs[imin]
        xbest = x[imin]

        self._initialize_log()

        # Iteration counter (at any point, iter is the number of fully executed
        # iterations so far).
        iteration = 0
        start_time = time.time()

        while True:
            iteration += 1

            print(
                f'iteration: {iteration}, cost: {cost_evaluations}, best: {fbest}'
            )

            stop_reason = self._check_stopping_criterion(
                start_time=start_time,
                iteration=iteration,
                cost_evaluations=cost_evaluations)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print("")
                break

            # Compute the inertia factor which we linearly decrease from 0.9 to
            # 0.4 from iteration = 0 to iteration = max_iterations.
            w = 0.4 + 0.5 * (1 - iteration / self._max_iterations)

            # Compute the velocities.
            for i, xi in enumerate(x):
                # Get the position and past best position of particle i.
                yi = y[i]

                # Get the previous position and velocity of particle i.
                xiprev = xprev[i]
                vi = v[i]

                # Compute the new velocity of particle i, composed of three
                # contributions.
                inertia = w * man.transp(xiprev, xi, vi)
                nostalgia = rnd.rand() * self._nostalgia * man.log(xi, yi)
                social = rnd.rand() * self._social * man.log(xi, xbest)

                v[i] = inertia + nostalgia + social

            # Backup the current swarm positions.
            xprev = list(x)

            if self._n_jobs > 0:
                with Pool(nodes=self._n_jobs) as p:
                    costs = p.map(objective, x)
            else:
                with Pool() as p:
                    costs = p.map(objective, x)

            # update the bests
            for i, fxi in enumerate(costs):
                # Update self-best if necessary.
                if fxi < fy[i]:
                    fy[i] = fxi
                    y[i] = x[i]
                    # Update global best if necessary.
                    if fy[i] < fbest:
                        fbest = fy[i]
                        xbest = x[i]

            # Compute new position of particle i.
            x = [man.retr(xi, vi) for xi, vi in zip(x, v)]

            cost_evaluations += self._population_size

        if multiple_answers:
            return xbest, x
        else:
            return xbest


class RandomAnswer(Solver):

    def __init__(
        self,
        populationsize=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._population_size = populationsize

    def solve(self, problem, x=None):
        """Return random params on a specified manifold
        """
        man = problem.manifold

        return man.rand()
