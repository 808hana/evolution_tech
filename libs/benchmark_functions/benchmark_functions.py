#!/usr/bin/env python3

import math
from scipy.stats import multivariate_normal
import numpy as np
import logging
from abc import ABC, abstractmethod
from . import functions_info_loader as fil


class BenchmarkFunction(ABC):
    def __init__(self, name, n_dimensions=2, opposite=False):
        if type(n_dimensions) != int:
            raise ValueError(f"The number of dimensions must be an int, found {n_dimensions}:{type(n_dimensions)}.")
        if n_dimensions == 0:
            raise ValueError(
                f"Functions can be created only using a number of dimensions greater than 0, found {n_dimensions}.")
        self._name = name
        self._n_dimensions = n_dimensions
        self.opposite = opposite
        self.parameters = []
        self.info = None

    def load_info(self, force_reload=False):
        if force_reload or self.info is None:
            self.info = fil.FunctionInfo(self._name)
        return self.info

    def __call__(self, point, validate=True):
        if validate:
            self._validate_point(point)
        if type(point) is fil.Optimum:
            point = point.position
        if self.opposite:
            return - self._evaluate(point)
        else:
            return self._evaluate(point)

    def gradient(self, point, validate=True):
        if validate:
            self._validate_point(point)
        if self.opposite:
            return - self._evaluate_gradient(point)
        else:
            return self._evaluate_gradient(point)

    # This returns always a numpy array
    def hessian(self, point, validate=True):
        if validate:
            self._validate_point(point)
        if self.opposite:
            return - self._evaluate_hessian(point)
        else:
            return self._evaluate_hessian(point)

    def _validate_point(self, point):
        if type(point) not in [tuple, list, np.ndarray, fil.Optimum]:
            raise ValueError(f"Functions can be evaluated only on tuple or lists of values, found {type(point)}")
        if len(point) != self._n_dimensions:
            raise ValueError(
                f"Function {self._name} declared for {self._n_dimensions} dimensions, asked to be evaluated on a point of {len(point)} dimensions")
        if not all(type(v) == float or type(v) == int or type(v) == np.float64 for v in point):
            idx = None
            for i in range(len(point)):
                if type(point[i]) not in [float, int]:
                    idx = i
                    break
            vs = [x for x in point]
            vs[idx] = str(vs[idx]) + "(" + str(type(vs[idx])) + ")"
            raise ValueError(f"Functions can only be evaluated on float or int values, passed {vs}")

    @abstractmethod
    def _evaluate(self, point):
        pass

    def _evaluate_gradient(self, point):
        raise NotImplementedError(f"Gradient of function {self.name} is not defined.")

    def _evaluate_hessian(self, point):
        raise NotImplementedError(f"Hessian of function {self.name} is not defined.")

    def name(self):
        return self._name

    def n_dimensions(self):
        return self._n_dimensions

    def minima(self):
        info = self.load_info()
        if self.opposite:
            optima = info.get_maxima(self._n_dimensions, self.parameters)
        else:
            optima = info.get_minima(self._n_dimensions, self.parameters)
        for o in optima:
            o.score = self(o.position)
        return optima

    # @return a dictionary {n_dimension: n_minima} with the number of known minima per each dimension. If dimension-invariant minima are known, a '*' char will be present in place of n_dimension
    def n_minima(self):
        info = self.load_info()
        if self.opposite:
            return info.get_number_maxima(self.parameters)
        else:
            return info.get_number_minima(self.parameters)

    # returns an object of type Optimum
    def minimum(self):
        minima = self.minima()
        if len(minima) == 0:
            return None
        pos = 0
        for i in range(len(minima))[1:]:
            if minima[i].score < minima[pos].score:
                pos = i
        return minima[pos]

    def maxima(self):
        info = self.load_info()
        if self.opposite:
            optima = info.get_minima(self._n_dimensions, self.parameters)
        else:
            optima = info.get_maxima(self._n_dimensions, self.parameters)
        for o in optima:
            o.score = self(o.position)
        return optima

    # @return a dictionary {n_dimension: n_maxima} with the number of known minima per each dimension. If dimension-invariant maxima are known, a '*' char will be present in place of n_dimension
    def n_maxima(self):
        info = self.load_info()
        if self.opposite:
            return info.get_number_minima(self.parameters)
        else:
            return info.get_number_maxima(self.parameters)

    # returns an object of type Optimum
    def maximum(self):
        maxima = self.maxima()
        if len(maxima) == 0:
            return None
        pos = 0
        for i in range(len(maxima))[1:]:
            if maxima[i][0] > maxima[pos][0]:
                pos = i
        return maxima[pos]

    def saddle_points(self):
        return self.load_info().get_saddles(self._n_dimensions, self.parameters)

    # @return a dictionary {n_dimension: n_saddle_points} with the number of known saddle points per each dimension. If dimension-invariant saddle points are known, a '*' char will be present in place of n_dimension
    def n_saddle_points(self):
        return self.load_info().get_number_saddles(self.parameters)

    def suggested_bounds(self):
        info = self.load_info()
        b = info.get_suggested_bounds(self.parameters)
        return ([b[0]] * self._n_dimensions, [b[1]] * self._n_dimensions)

    def reference(self):
        return self.load_info().get_reference()

    def summary(self):
        return self.load_info().get_summary()

    def dev_comment(self):
        return self.load_info().get_dev_comment()

    def definition(self):
        return self.load_info().get_definition()

    def _validate_bonds(self, bounds):
        if len(bounds) != 2:
            raise ValueError(f"Boundaries must be of the ([lb_0, lb_1, ...], [ub_0, ub_1, ...]) passed: {bounds}")
        if len(bounds[0]) != self._n_dimensions or len(bounds[1]) != self._n_dimensions:
            raise ValueError(f"Expected boundaries of {self._n_dimensions} dimensions, passed: {bounds}")

    def show(self, bounds=None, asHeatMap=False, resolution=50, showPoints=[]):
        if self._n_dimensions > 2:
            raise ValueError(f"Only functions defined in 1 or 2 dimensions can be visualised (N={self._n_dimensions})")
        if self._n_dimensions == 1 and asHeatMap:
            raise ValueError(
                f"Only functions defined in 2 dimensions can be visualised as heatmap (N={self._n_dimensions})")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.error("In order to show the function the matplotlib module is required.")
            return
        if bounds is None:
            bounds_lower, bounds_upper = self.suggested_bounds()
        else:
            self._validate_bonds(bounds)
            bounds_lower, bounds_upper = bounds
        if len(showPoints) > 0:
            for p in showPoints:
                if len(p) != self._n_dimensions:
                    raise ValueError(f"Points to plot must be all of {self._n_dimensions} dimensions, passed: {p}")

        def in_bounds(p):
            for i in range(len(p)):
                if p[i] < bounds_lower[i] or p[i] > bounds_upper[i]:
                    return False
            return True

        showPoints = [x for x in showPoints if in_bounds(x)]

        x = np.linspace(bounds_lower[0], bounds_upper[0], resolution)
        if self._n_dimensions > 1:
            y = np.linspace(bounds_lower[1], bounds_upper[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = np.asarray([[self((X[i][j], Y[i][j])) for j in range(len(X[i]))] for i in range(len(X))])
        fig = plt.figure()
        #fig.canvas.set_window_title('Benchmark Function: '+self._name)
        fig.suptitle(self._name)
        if asHeatMap:
            plt.contour(x, y, Z, 15, linewidths=0.5, colors='k')  # hight lines
            plt.contourf(x, y, Z, 15, cmap='viridis', vmin=Z.min(), vmax=Z.max())  # heat map
            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar()
            cbar.set_label('z')
            if len(showPoints) > 0:  # plot points
                xdata = [x[0] for x in showPoints]
                ydata = [x[1] for x in showPoints]
                plt.scatter(xdata, ydata, c='r')
        elif self._n_dimensions == 1:
            y = np.asarray([self([v]) for v in x])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(x, y)
            if len(showPoints) > 0:  # plot points
                plt.scatter(showPoints, [self(x) for x in showPoints], c='r')
        else:
            ax = plt.axes(projection='3d')
            if len(showPoints) > 0:  # plot points
                xdata = [x[0] for x in showPoints]
                ydata = [x[1] for x in showPoints]
                zdata = [self(x) for x in showPoints]
                ax.scatter3D(xdata, ydata, zdata, c='r')
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.7)
            else:
                ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.view_init(60, 35)
            ax.set_zlabel('z')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        plt.show()

    '''
    By default, only isolated strict local minima are consider valid (i.e. \exists \epsilon>0\mid f(x^*)<f(x) \forall x\in B(x^*,\epsilon)).
    If the flag strict is set to False, a potential local minimum is considered valid if it is located in a plateau (i.e. f(x^*)<f(x)).
    An approximation is used to validate the potential minimum x^*, assessing the immediate surrounding of x^* for points of lower score, according a certain threshold. 
    Both the radius factor (i.e. the radius relaive to the boundaries size) and the threshold can be independently set with two optional parameters.
    This functions only assess the validity of the given point, it does NOT perform any serch and it is too expensive to iterate it in an attempt of searching a local minimum.
    Please note that this function is only intended as a non-rigorous validation check. It may work well in many cases, however the use of a small (but not infinitesimal) radius, a finite number of tests around the solution and numerical problems are all factors that may result in both false positive and false negative outcomes.
    @return a tuple (response, point, message) where response is boolean and, if it is False, 'message' contains information on why the candidate solution is not a valid local minimum, whilst 'point' will be a better local minimum
    '''

    def testLocalMinimum(self, potential_minimum, radius_factor=1e-10, score_threshold=1e-6, n_tests=int(1e5),
                         strict=True):
        if score_threshold < 0.0:
            raise ValueError(f"Score threshold must be a non-negative number, passed {score_threshold}")
        if radius_factor <= 0.0:
            raise ValueError(f"Radius factor must be a positive number, passed {radius_factor}")
        if not type(n_tests) is int or n_tests <= 0:
            raise ValueError(f"The number of tests must be a positive number, passed {n_tests}")
        self._validate_point(potential_minimum)
        lb, ub = self.suggested_bounds()
        r = np.linalg.norm(np.array(ub) - np.array(lb)) * radius_factor  # TODO gotta be sure it doesn't underflow
        pms = self(potential_minimum, validate=False)
        for _ in range(n_tests):
            # sample points using the Muller, Marsaglia (‘Normalised Gaussians’) method
            a = [np.random.normal(0, 1) for _ in range(self._n_dimensions)]
            an = np.linalg.norm(a)
            p = a / an * np.random.uniform(r * 1e-2, r)
            p = [p[i] + potential_minimum[i] for i in range(self._n_dimensions)]
            ps = self(p, validate=False)
            diff = ps - pms
            if strict:
                if score_threshold - diff <= 0.0:
                    message = f"Point {p} has a distance of {np.linalg.norm(np.array(p) - np.array(potential_minimum))} from the candidate point {potential_minimum} and it has a lower or equals value than it ({ps}<={pms})"
                    return (False, p, message)
            else:
                if score_threshold - diff < 0.0:
                    message = f"Point {p} has a distance of {np.linalg.norm(np.array(p) - np.array(potential_minimum))} from the candidate point {potential_minimum} and it has a lower value than it ({ps}<{pms})"
                    return (False, p, message)
        if score_threshold == 0.0:
            return (True, None, f'Point {potential_minimum} is probably a local minimum for {self._name}!')
        else:
            return (True, None,
                    f'Point {potential_minimum} is probably a local minimum for {self._name} under the {score_threshold} threshold!')

    def minimum_grid_search(self, bounds, n_edge_points=100,
                            score_threshold=0.0):  # the number of points sampled is (n_edge_points+1)^N
        if bounds is None:
            bounds_lower, bounds_upper = self.suggested_bounds()
        else:
            self._validate_bonds(bounds)
            bounds_lower, bounds_upper = bounds
        # ok, let's try to do this in place, shall we?
        MAX_MIN_POINTS = 100
        min_points = []  # it is a list because it is possible to find a plateau
        point = [bounds_lower[i] for i in range(self._n_dimensions)]
        idx = 0
        while True:
            v = self(point, validate=False)
            if len(min_points) == 0 or v - min_points[0][1] < score_threshold:
                min_points = [(list(point), v)]
            elif v == min_points[0][1] and len(min_points) < MAX_MIN_POINTS:
                min_points += [(list(point), v)]
            point[idx] += (bounds_upper[idx] - bounds_lower[idx]) / n_edge_points
            if point[idx] > bounds_upper[idx]:
                if idx == self._n_dimensions - 1:
                    break
                while idx < self._n_dimensions and point[idx] >= bounds_upper[idx]:
                    point[idx] = bounds_lower[idx]
                    idx += 1
                if idx >= self._n_dimensions:
                    break
                point[idx] += (bounds_upper[idx] - bounds_lower[idx]) / n_edge_points
                idx = 0
        if len(min_points) == 1:
            return min_points[0]
        return min_points

    def minimum_random_search(self, bounds, n_samples=int(1e7), score_threshold=0.0):
        if bounds is None:
            bounds_lower, bounds_upper = self.suggested_bounds()
        else:
            self._validate_bonds(bounds)
            bounds_lower, bounds_upper = bounds
        MAX_MIN_POINTS = 100
        min_points = []  # it is a list because it is possible to find a plateau
        for _ in range(n_samples):
            point = [np.random.uniform(bounds_lower[i], bounds_upper[i]) for i in range(self._n_dimensions)]
            v = self(point, validate=False)
            if len(min_points) == 0 or v - min_points[0][1] < score_threshold:
                min_points = [(list(point), v)]
            elif v == min_points[0][1] and len(min_points) < MAX_MIN_POINTS:
                min_points += [(list(point), v)]
        if len(min_points) == 1:
            return min_points[0]
        return min_points


'''
Continuous, convex and unimodal.
'''


class Hypersphere(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Hypersphere", n_dimensions, opposite)

    def _evaluate(self, point):
        return sum([pow(x, 2) for x in point])

    def _evaluate_gradient(self, point):
        return [2.0 * x for x in point]

    def _evaluate_hessian(self, point):
        H = np.zeros((self._n_dimensions, self._n_dimensions))
        np.fill_diagonal(H, 2.0)
        return H


'''
Continuous, convex and unimodal.
'''


class Hyperellipsoid(BenchmarkFunction):  # rotated hyperellipsoid
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Hyperellipsoid", n_dimensions, opposite)

    def _evaluate(self, point):
        ret = 0.0
        for i in range(self._n_dimensions):
            for j in range(i + 1):
                ret += pow(point[j], 2)
        return ret


'''
Continuous, non-convex and multimodal.
'''


class Rosenbrock(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Rosenbrock", n_dimensions, opposite)

    def _evaluate(self, point):
        s = 0.0
        for i in range(len(point) - 1):
            s += 100 * pow(point[i + 1] - pow(point[i], 2), 2) + pow(1.0 - point[i], 2)
        return s


'''
Continuous, non-convex and (highly) multimodal. 
Location of the minima are regularly distributed.
'''


class Rastrigin(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Rastrigin", n_dimensions, opposite)

    def _evaluate(self, point):
        ret = sum([pow(p, 2) - 10.0 * math.cos(2.0 * math.pi * p) for p in point]) + 10.0 * len(point)
        return ret
    # def _evaluate_derivative(self, point):
    # 	return sum([2.0*p + 20.0*math.pi*math.sin(2.0*math.pi*p) for p in point])
    # def _evaluate_second_derivative(self, point):
    # 	return sum([2.0 + 40.0*pow(math.pi,2)*math.cos(2.0*math.pi*p) for p in point])


'''
Continuous, non-convex and (highly) multimodal. 
Location of the minima are geometrical distant.
'''


class Schwefel(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Schwefel", n_dimensions, opposite)

    def _evaluate(self, point):
        ret = 418.9829 * self.n_dimensions() - sum([p * math.sin(math.sqrt(abs(p))) for p in point])
        return ret
    # def _evaluate_derivative(self, point):
    # 	if point==[0.0]*len(point):
    # 		return 0.0
    # 	else:
    # 		return sum([-pow(p,2)*math.cos(math.sqrt(abs(p)))/(2.0*pow(abs(p),3.0/2.0)) - math.sin(math.sqrt(abs(p))) for p in point if p!=0.0])

'''
Continuous, non-convex and (highly) multimodal. 
The suggested behaviour is zoom-dependent: 
    - [zoom=0] general overview [-600<= x_i <= 600] suggests convex function;
    - [zoom=1] medium-scale view [-10 <= x_i <= 10] suggests existence of local optima;
    - [zoom=2] zoom on the details [-5 <= x_i <= 5] reveal complex structure of numerous local optima;
'''


class Griewank(BenchmarkFunction):
    def __init__(self, n_dimensions=2, zoom=0, opposite=False):
        if zoom not in [0, 1, 2]:
            raise ValueError("Griewank function defined with a zoom level not in [0,1,2]")
        super().__init__("Griewank", n_dimensions, opposite)
        self.parameters = [("zoom", zoom)]

    def getName(self):
        return "Griewank"

    def _evaluate(self, point):
        part1 = 0.0
        part2 = 1.0
        for i in range(len(point)):
            part1 += pow(point[i], 2)
            part2 *= math.cos(point[i] / math.sqrt(i + 1))
        ret = 1.0 + part1 / 4000.0 - part2
        return ret


'''
Continuous, non-convex and multimodal.
Clear global minimum at the center surrounded by many symmetrical local minima.
'''


class Ackley(BenchmarkFunction):
    def __init__(self, n_dimensions=2, a=20, b=.2, c=2.0 * math.pi, opposite=False):
        super().__init__("Ackley", n_dimensions, opposite)
        self.a = a
        self.b = b
        self.c = c

    def _evaluate(self, point):
        part1 = 0.0
        part2 = 0.0
        for i in range(len(point)):
            part1 += pow(point[i], 2)
            part2 += math.cos(self.c * point[i])
        ret = -self.a * math.exp(-self.b * math.sqrt(part1 / len(point))) - math.exp(
            part2 / len(point)) + self.a + math.exp(1.0)
        return ret


'''
Continuous, non-convex and (highly) multimodal. 
Contains n! local minimum. 
'''


class Michalewicz(BenchmarkFunction):
    def __init__(self, n_dimensions=2, m=10, opposite=False):
        super().__init__("Michalewicz", n_dimensions, opposite)
        self.m = m

    def _evaluate(self, point):
        s = 0.0
        for i in range(len(point)):
            s += math.sin(point[i]) * pow(math.sin((i + 1) * pow(point[i], 2) / math.pi), 2 * self.m)
        return -s  # it's not an error


'''
Non-convex, contains multiple asymmetrical local optima
'''


class EggHolder(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Egg Holder", n_dimensions, opposite)

    def _evaluate(self, point):
        s = 0.0
        for i in range(len(point) - 1):
            s += (point[i + 1] + 47) * math.sin(math.sqrt(abs(point[i + 1] + 47.0 + point[i] / 2.0))) + point[
                i] * math.sin(math.sqrt(abs(point[i] - (point[i + 1] + 47.0))))
        return -s


'''
Mutlimodal function with local optima regions of different depths
'''


class Keane(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Keane", n_dimensions, opposite)

    # validate the point according the Keane's function conditions
    def validate(self, point):
        p = 1.0
        for x in point:
            p *= x
        if 0.75 > p:
            raise ValueError(f"Product condiction violated on the Keane's function (0.75>{p})")
        if sum(point) > 7.5 * len(point):
            raise ValueError(f"Sum condiction violated on the Keane's function ({7.5 * len(point)}<{sum(point)})")

    def _evaluate(self, point):
        if sum(point) == 0:
            return 0.0
        part0 = 1.0
        for x in point:
            part0 *= np.cos(x) ** 2
        part1 = abs(sum([np.cos(x) ** 4 for x in point]) - 2.0 * part0)
        part2 = np.sqrt(sum([(i + 1) * point[i]**2 for i in range(len(point))]))
        return -part1 / part2


'''
Highly multimodal symmetric function 
'''


class Rana(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Rana", n_dimensions, opposite)

    def _evaluate(self, point):
        s = 0.0
        for i in range(len(point) - 1):
            p1 = point[i] * math.cos(math.sqrt(abs(point[i + 1] + point[i] + 1.0)))
            p2 = math.sin(math.sqrt(abs(point[i + 1] - point[i] + 1.0)))
            p3 = (1.0 + point[i + 1]) * math.sin(math.sqrt(abs(point[i + 1] + point[i] + 1.0)))
            p4 = math.cos(math.sqrt(abs(point[i + 1] - point[i] + 1.0)))
            s += p1 * p2 + p3 * p4
        return s


'''
Continuous, unimodal, mostly a plateau with global minimum in a small central area.
It's defined only for 2 dimensions.
'''


class Easom(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Easom", 2, opposite)

    def _evaluate(self, point):
        ret = -math.cos(point[0]) * math.cos(point[1]) * math.exp(
            -pow(point[0] - math.pi, 2) - pow(point[1] - math.pi, 2))
        return ret


'''
Multimodal, "stairs"-like function, with multiple plateau at different levels
'''


class DeJong3(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("De Jong 3", n_dimensions, opposite)

    def _evaluate(self, point):
        ret = sum([math.floor(x) for x in point])
        return ret

    # optima are generated programmatically
    def _get_plateauy_centers(self, otype):
        lb, ub = self.suggested_bounds()
        plateau_centers = []
        point = [np.sign(x) * np.ceil(np.abs(x)) for x in lb]
        idx = 0
        while True:
            p = [(i + (i + 1)) * 0.5 for i in point]
            o = fil.Optimum(['_', p], otype)
            plateau_centers += [o]
            point[idx] += 1
            if point[idx] > ub[idx]:
                while idx < self._n_dimensions and point[idx] + 1 > ub[idx]:
                    # reset the current value and move one step to the right
                    point[idx] = np.sign(lb[idx]) * np.ceil(np.abs(lb[idx]))
                    idx += 1
                if idx >= self._n_dimensions or (idx == self._n_dimensions - 1 and point[idx] + 1 > ub[idx]):
                    # my work here is done
                    break
                point[idx] += 1
                idx = 0
        return plateau_centers

    def minima(self):
        optima = self._get_plateauy_centers(otype='minima')
        for o in optima:
            o.score = self(o.position)
        return optima

    def minimum(self):  # this one is more efficient
        lb, ub = self.suggested_bounds()
        if self.opposite:
            optimum = [np.sign(ub[i]) * (np.ceil(np.abs(ub[i])) - .5) for i in range(self._n_dimensions)]
        else:
            optimum = [np.sign(lb[i]) * (np.ceil(np.abs(lb[i])) - .5) for i in range(self._n_dimensions)]
        o = fil.Optimum(['_', optimum], 'minima')
        o.score = self(o.position)
        return o

    def maxima(self):
        optima = self._get_plateauy_centers(otype='maxima')
        for o in optima:
            o.score = self(o.position)
        return optima

    def maximum(self):  # this one is more efficient
        lb, ub = self.suggested_bounds()
        if self.opposite:
            optimum = [np.sign(lb[i]) * (np.ceil(np.abs(lb[i])) - .5) for i in range(self._n_dimensions)]
        else:
            optimum = [np.sign(ub[i]) * (np.ceil(np.abs(ub[i])) - .5) for i in range(self._n_dimensions)]
        o = fil.Optimum(['_', optimum], 'maxima')
        o.score = self(o.position)
        return o

    def _evaluate_gradient(self, point):
        res = []
        for i in range(self._n_dimensions):
            if point[i].is_integer():
                res += [None]  # the partial derivative is not defined on a corner
            else:
                res += [0.0]
        return res


'''
Continuous, multimodal, multiple symmetric local optima with narrow basins on a plateau
It's defined only for 2 dimensions.
'''


class DeJong5(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("De Jong 5", 2, opposite)
        self.A = [
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]

    def _evaluate(self, point):
        ret = pow(0.002 + sum(
            [1.0 / (i + 1.0 + pow(point[0] - self.A[0][i], 6) + pow(point[1] - self.A[1][i], 6)) for i in range(25)]),
                  -1)
        return ret


'''
Continuous, multimodal with an asymmetrical hight slope and global minimum on a plateau.
It's defined only for 2 dimensions.
'''


class GoldsteinAndPrice(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Goldstein and Price", 2, opposite)

    def _evaluate(self, point):
        a = 1.0 + pow(point[0] + point[1] + 1.0, 2) * (
                    19.0 - 14.0 * point[0] + 3.0 * pow(point[0], 2) - 14.0 * point[1] + 6.0 * point[0] * point[
                1] + 3.0 * pow(point[1], 2))
        b = 30.0 + pow(2 * point[0] - 3.0 * point[1], 2) * (
                    18.0 - 32.0 * point[0] + 12.0 * pow(point[0], 2) + 48.0 * point[1] - 36.0 * point[0] * point[
                1] + 27.0 * pow(point[1], 2))
        return a * b


'''
(logaritmic variant of Goldstein and Price) continuous, with multiple asymmetrical slopes and global minimum near local optima.
It's defined only for 2 dimensions.
'''


class PichenyGoldsteinAndPrice(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Picheny, Goldstein and Price", 2, opposite)

    def _evaluate(self, point):
        x1 = 4.0 * point[0] - 2.0
        x2 = 4.0 * point[1] - 2.0
        a = 1.0 + pow(x1 + x2 + 1.0, 2) * (
                    19.0 - 14.0 * x1 + 3.0 * pow(x1, 2) - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * pow(x2, 2))
        b = 30.0 + pow(2 * x1 - 3.0 * x2, 2) * (
                    18.0 - 32.0 * x1 + 12.0 * pow(x1, 2) + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * pow(x2, 2))
        ret = 1.0 / 2.427 * (math.log(a * b) - 8.693)
        return ret


'''
continuous, multimodal, with optima displaced in a symmetric way.
'''


class StyblinskiTang(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Styblinski and Tang", n_dimensions, opposite)

    def _evaluate(self, point):
        ret = sum([pow(x, 4) - 16.0 * pow(x, 2) + 5.0 * x for x in point]) / 2.0
        return ret


'''
Continuous, unimodal, uneven slopes on the sides
It's defined only for 2 dimensions.
'''


class McCormick(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("McCormick", 2, opposite)

    def _evaluate(self, point):
        ret = math.sin(point[0] + point[1]) + pow(point[0] - point[1], 2) - 1.5 * point[0] + 2.5 * point[1] + 1.0
        return ret


class MartinGaddy(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Martin and Gaddy", 2, opposite)

    def _evaluate(self, point):
        ret = pow(point[0] - point[1], 2) + pow((point[0] + point[1] - 10.0) / 3.0, 2)
        return ret


class Schaffer2(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Schaffer 2", 2, opposite)

    def _evaluate(self, point):
        tmp = pow(point[0], 2) + pow(point[1], 2)
        ret = 0.5 + (pow(math.sin(math.sqrt(tmp)), 2) - 0.5) / pow(1.0 + 0.001 * tmp, 2)
        return ret


class Himmelblau(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Himmelblau", 2, opposite)

    def _evaluate(self, point):
        ret = pow(pow(point[0], 2) + point[1] - 11, 2) + pow(point[0] + pow(point[1], 2) - 7, 2)
        return ret


class PitsAndHoles(BenchmarkFunction):
    def __init__(self, opposite=False):
        super().__init__("Pits and Holes", 2, opposite)
        self.mu = [[0, 0], [20, 0], [0, 20], [-20, 0], [0, -20], [10, 10], [-10, -10], [-10, 10], [10, -10]]
        self.c = [10.5, 14.0, 16.0, 12.0, 9.0, 0.1, 0.2, 0.25, 0.17]
        self.v = [2.0, 2.5, 2.7, 2.5, 2.3, 0.05, 0.3, 0.24, 0.23]

    def _get_covariance_matrix(self, idx):
        return [[self.c[idx], 0], [0, self.c[idx]]]

    def _evaluate(self, point):
        v = 0
        for i in range(len(self.mu)):
            v += multivariate_normal.pdf(point, mean=self.mu[i], cov=self._get_covariance_matrix(i)) * self.v[i]
        return -v


'''
Continuous, non-convex, and multimodal.
'''


class Levy(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Levy", n_dimensions, opposite)

    def _evaluate(self, point):
        w = [1 + (x - 1) / 4 for x in point]

        term1 = pow(np.sin(np.pi * w[0]), 2)
        term_sum = sum(pow(w[i] - 1, 2) * (1 + 10 * pow(np.sin(np.pi * w[i] + 1), 2)) for i in range(len(w) - 1))
        term_last = pow(w[-1] - 1, 2) * (1 + pow(np.sin(2 * np.pi * w[-1]), 2))

        return term1 + term_sum + term_last


'''
Continuous, non-convex, and multimodal.
'''


class Shubert(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Shubert", n_dimensions, opposite)

    def _evaluate(self, point):
        result = 1
        for xi in point:
            inner_sum = sum(j * np.cos((j + 1) * xi + j) for j in range(1, 6))
            result *= inner_sum  # Product of sums
        return result


'''
Continuous, non-convex, and multimodal.
It's defined only for 2 dimensions. -> modified for n-dimensions
'''


class DropWave(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Drop Wave", n_dimensions, opposite)

    def _evaluate(self, point):
        norm_sq = np.sum(np.square(point))
        norm_val = np.sqrt(norm_sq)
        numerator = 1 + np.cos(12 * norm_val)
        denominator = 0.5 * norm_sq + 2
        return -numerator / denominator


'''
Continuous, multimodal, and challenging function for optimization algorithms.
It's defined only for 2 dimensions.
'''


class CrossInTray(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        if n_dimensions != 2:
            raise ValueError("Cross-in-Tray function is only defined for 2 dimensions.")
        super().__init__("Cross In Tray", n_dimensions, opposite)

    def _evaluate(self, point):
        if len(point) != 2:
            raise ValueError("Cross-in-Tray function requires exactly 2 input values.")
        x, y = point
        term1 = np.sin(x) * np.sin(y)
        term2 = np.exp(np.abs(100 - np.sqrt(x ** 2 + y ** 2) / np.pi))
        result = -0.0001 * (np.abs(term1 * term2) + 1) ** 0.1
        return result


'''
Continuous, multimodal, and non-convex.
It's defined only for 2 dimensions.
'''


class Bohachevsky(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        if n_dimensions != 2:
            raise ValueError("Bohachevsky function is defined only for 2 dimensions.")
        super().__init__("Bohachevsky", n_dimensions, opposite)

    def _evaluate(self, point):
        if len(point) != 2:
            raise ValueError("Bohachevsky function requires a 2D input.")

        x, y = point
        return x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x) - 0.4 * np.cos(4 * np.pi * y) + 0.7


# supr cupr evolutionary benchmarking functions

'''
Continuous, non-convex, and multimodal.

In the late 19th century, Karl Weierstrass invented a fractal-like function 
that was decried as nothing less than a “deplorable evil.” 
In time, it would transform the foundations of mathematics. 

self._n_dimensions
'''


class Weierstrass(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False, a=0.5, b=3, K=20):
        super().__init__("Weierstrass", n_dimensions, opposite)
        self.a = a
        self.b = b
        self.K = K

    def _evaluate(self, point):
        # Ensure 'point' is a numpy array if needed
        point = np.array(point, dtype=float)
        n = self.n_dimensions()

        inner_sum = 0.0
        for i in range(n):
            x_i = point[i]
            # sum_{k=0}^{K} a^k cos(2π b^k (x_i + 0.5))
            for k in range(self.K + 1):
                inner_sum += (self.a ** k) * np.cos(2.0 * np.pi * (self.b ** k) * (x_i + 0.5))

        outer_sum = 0.0
        # sum_{k=0}^{K} a^k cos(π b^k)
        for k in range(self.K + 1):
            outer_sum += (self.a ** k) * np.cos(np.pi * (self.b ** k))

        return inner_sum - n * outer_sum


'''
Continuous, non-convex, and multimodal.
'''


class Salomon(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Salomon", n_dimensions, opposite)

    def _evaluate(self, point):
        point = np.array(point, dtype=float)
        r = np.sqrt(np.sum(point ** 2))
        return 1.0 - np.cos(2.0 * np.pi * r) + 0.1 * r


'''
Continuous, non-convex, and multimodal.
'''


class Katsuura(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False, K=32):
        super().__init__("Katsuura", n_dimensions, opposite)
        self.K = K  # Maximum power of 2

    def _evaluate(self, point):
        point = np.array(point, dtype=float)
        n = self.n_dimensions()

        product_term = 1.0
        for i in range(n):
            x_i = point[i]
            sum_inner = 0.0
            for k in range(1, self.K + 1):
                t = (2.0 ** k) * x_i
                sum_inner += abs(t - np.floor(t) - 0.5) / (2.0 ** k)

            # Multiply the product:
            product_term *= (1.0 + (i + 1) * sum_inner) ** (10.0 / (n ** 2))

        return product_term - 10.0 / (n ** 2)


'''
Continuous, non-convex, and multimodal.
'''


class Trid(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Trid", n_dimensions, opposite)

    def _evaluate(self, point):
        point = np.array(point, dtype=float)
        n = self.n_dimensions()

        square_sum = 0.0
        for i in range(n):
            square_sum += (point[i] - 1.0) ** 2

        cross_sum = 0.0
        for i in range(1, n):
            cross_sum += point[i] * point[i - 1]

        return square_sum - cross_sum


'''
Continuous, non-convex, and multimodal.
'''


class Whitley(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Whitley", n_dimensions, opposite)

    def _evaluate(self, point):
        point = np.array(point, dtype=float)
        n = self.n_dimensions()

        val = 0.0
        for i in range(n):
            for j in range(n):
                # a = point[i]
                # b = point[j]
                # t1 = (a - 1.0) ** 2 * (1.0 + np.sin(3.0 * b)) ** 2
                # t2 = (b - 1.0) ** 2 * (1.0 + np.sin(3.0 * a)) ** 2
                # t3 = np.cos((a - 1.0) * (b - 1.0))
                # val += (t1 + t2) / 4000.0 + 1.0 - t3

                temp = 100.0 * (((point[i] ** 2.0) - point[j])**2) + (1.0 - point[j]) ** 2.0
                val += (float(temp ** 2.0) / 4000.0) - np.cos(temp) + 1.0

        return val


'''
Continuous, non-convex, and multimodal.
'''


class Alpine1(BenchmarkFunction):
    def __init__(self, n_dimensions=2, opposite=False):
        super().__init__("Alpine1", n_dimensions, opposite)

    def _evaluate(self, point):
        x = np.array(point)
        values = x * np.sin(x) + 0.1 * x
        return np.sum(np.abs(values))


# class Shekel(BenchmarkFunction):
# 	def __init__(self,n_dimensions,m=10, opposite=False):
# 		super().__init__("Shekel", n_dimensions, opposite)
# 		self.parameters=[('m',m)]
# 		self.m=m
# 		if n_dimensions==2:
# 			self.c=(0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5)
# 			self.A=[[-32, -16,  0,  16,  32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32], 
# 							[-32, -32, -32,-32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
# 		elif n_dimensions==4:
# 			self.c=(0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5)
# 			self.A=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]]
# 		elif n_dimensions==10:
# 			self.c=(0.806, 0.517, 0.10, 0.908, 0.965, 0.669, 0.524, 0.902, 0.531, 0.876, 0.462,
# 							0.491, 0.463, 0.714, 0.352, 0.869, 0.813, 0.811, 0.828, 0.964, 0.789,
# 							0.360, 0.369, 0.992, 0.332, 0.817, 0.632, 0.883, 0.608, 0.326)
# 			self.A=[[9.681,0.667,4.783,9.095,3.517,9.325,6.544,0.211,5.122,2.020],
# 							[9.400,2.041,3.788,7.931,2.882,2.672,3.568,1.284,7.033,7.374],
# 							[8.025,9.152,5.114,7.621,4.564,4.711,2.996,6.126,0.734,4.982],
# 							[2.196,0.415,5.649,6.979,9.510,9.166,6.304,6.054,9.377,1.426],
# 							[8.074,8.777,3.467,1.863,6.708,6.349,4.534,0.276,7.633,1.567],
# 							[7.650,5.658,0.720,2.764,3.278,5.283,7.474,6.274,1.409,8.208],
# 							[1.256,3.605,8.623,6.905,0.584,8.133,6.071,6.888,4.187,5.448],
# 							[8.314,2.261,4.224,1.781,4.124,0.932,8.129,8.658,1.208,5.762],
# 							[0.226,8.858,1.420,0.945,1.622,4.698,6.228,9.096,0.972,7.637],
# 							[305,2.228,1.242,5.928,9.133,1.826,4.060,5.204,8.713,8.247],
# 							[0.652,7.027,0.508,4.876,8.807,4.632,5.808,6.937,3.291,7.016],
# 							[2.699,3.516,5.874,4.119,4.461,7.496,8.817,0.690,6.593,9.789],
# 							[8.327,3.897,2.017,9.570,9.825,1.150,1.395,3.885,6.354,0.109],
# 							[2.132,7.006,7.136,2.641,1.882,5.943,7.273,7.691,2.880,0.564],
# 							[4.707,5.579,4.080,0.581,9.698,8.542,8.077,8.515,9.231,4.670],
# 							[8.304,7.559,8.567,0.322,7.128,8.392,1.472,8.524,2.277,7.826],
# 							[8.632,4.409,4.832,5.768,7.050,6.715,1.711,4.323,4.405,4.591],
# 							[4.887,9.112,0.170,8.967,9.693,9.867,7.508,7.770,8.382,6.740],
# 							[2.440,6.686,4.299,1.007,7.008,1.427,9.398,8.480,9.950,1.675],
# 							[6.306,8.583,6.084,1.138,4.350,3.134,7.853,6.061,7.457,2.258],
# 							[0.652,2.343,1.370,0.821,1.310,1.063,0.689,8.819,8.833,9.070],
# 							[5.558,1.272,5.756,9.857,2.279,2.764,1.284,1.677,1.244,1.234],
# 							[3.352,7.549,9.817,9.437,8.687,4.167,2.570,6.540,0.228,0.027],
# 							[8.798,0.880,2.370,0.168,1.701,3.680,1.231,2.390,2.499,0.064],
# 							[1.460,8.057,1.336,7.217,7.914,3.615,9.981,9.198,5.292,1.224],
# 							[0.432,8.645,8.774,0.249,8.081,7.461,4.416,0.652,4.002,4.644],
# 							[0.679,2.800,5.523,3.049,2.968,7.225,6.730,4.199,9.614,9.229],
# 							[4.263,1.074,7.286,5.599,8.291,5.200,9.214,8.272,4.398,4.506],
# 							[9.496,4.830,3.150,8.270,5.079,1.231,5.731,9.494,1.883,9.732],
# 							[4.138,2.562,2.532,9.661,5.611,5.500,6.886,2.341,9.699,6.500]]
# 		else:
# 			raise ValueError("The Shekel function is only defined for 2,4 or 10 dimensions")
# 	def _evaluate(self,point):
# 		return sum([pow(self.c[i] + sum([pow(point[j] - self.A[j][i],2) for j in range(self._n_dimensions)]),-1) for i in range(self.m)])
# 		n_dimensions=len(point)
# 		if n_dimensions==2:
# 			s=sum([pow(j + pow(point[0] - self.A[0][j],9) + pow(point[1] - self.A[1][j],6),-1) for j in range(24)]) # in the formal equation it should be ^6 in both cases, instead it works only if it's ^9 and ^6
# 			ret = pow(1.0/500.0 + s,-1)
# 		elif n_dimensions==4:
# 			ret = -sum([pow(sum([pow(point[j] - self.A[i][j],2) for j in range(self._n_dimensions)]) + self.c[i],-1) for i in range(self.m)])
# 		elif n_dimensions==10:
# 			ret = sum([pow(sum([pow(point[j] - self.A[i][j],2) for j in range(self._n_dimensions)]) + self.c[i],-1) for i in range(30)])
# 		else:
# 			raise ValueError("The Shekel function is only defined for 2,4 or 10 dimensions")
# 		return ret


def _validateAllMinima():
    f = DeJong5()
    f.show()
    mm = f.minima()
    for m in mm:
        res = f.testLocalMinimum(m[1])
        if not res[0]:
            print(res)

    # for n in range(1,11):
    # 	f=DeJong3(n_dimensions=n)
    # 	mm = f.minima()
    # 	for m in mm:
    # 		res = f.testLocalMinimum(m[1])
    # 		if not res[0]:
    # 			print(res)


if __name__ == "__main__":
    _validateAllMinima()
