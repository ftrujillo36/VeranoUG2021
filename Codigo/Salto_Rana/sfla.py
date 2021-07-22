import numpy as np
import matplotlib.pyplot as plt


class sflaSolver:
    def __init__(self, frogs_no, mplx_no, w1, w2, sigma) -> None:
        # Inicializar los parámetros del algoritmo
        self.frogs_no = frogs_no
        self.mplx_no = mplx_no
        self.w1 = w1
        self.w2 = w2
        self.sigma = sigma

    def opt_func(self, frog, ffnc='exp'):
        """Fitness function. Evaluates the fitness of the given frog

        Arguments:
            frog {np.ndarray} -- An individual value or frog

        Returns:
            float -- The output value or fitness of the frog
        """
        # Find the distances between the frog and registered obstacles
        distances = np.array(list(map(np.linalg.norm, self.obstacles - frog)))
        norm = np.amin(distances) if distances.size else 0
        # Fitness function
        if ffnc == 'exp':
            output = self.w1 * np.exp(-norm) + self.w2 * \
                np.linalg.norm(self.target - frog)
        elif ffnc == 'rtnl':
            output = self.w1 * (1 / norm) + self.w2 * \
                np.linalg.norm(self.target - frog)
        else:
            output = -1
            print("Unknown fitness function")

        return output

    def gen_frogs(self, n_frogs):
        """Generates a random frog using a gaussian normal distribution around the position

        Arguments:
            n_frogs {int} -- Number of frogs

        Returns:
            numpy.ndarray -- A frogs x dimension array
        """

        # Create random positions close to the current position
        xi = np.random.normal(self.start[0], self.sigma, n_frogs)
        yi = np.random.normal(self.start[1], self.sigma, n_frogs)
        frogs = np.stack((xi, yi), axis=1)
        return frogs

    def sort_frogs(self, frogs, mplx_no):
        """Sorts the frogs in decending order of fitness by the given function.

        Arguments:
            frogs {numpy.ndarray} -- Frogs to be sorted
            mplx_no {int} -- Number of memeplexes, when divides frog number should return an integer otherwise frogs will be skipped

        Returns:
            numpy.ndarray -- A memeplexes x frogs/memeplexes array of indices, [0, 0] will be the greatest frog
        """

        # Find fitness of each frog
        fitness = np.array(list(map(self.opt_func, frogs)))
        # Sort the indices in decending order by fitness
        sorted_fitness = np.argsort(fitness)
        # Empty holder for memeplexes
        memeplexes = np.zeros((mplx_no, int(frogs.shape[0]/mplx_no)))
        # Sort into memeplexes
        for j in range(memeplexes.shape[1]):
            for i in range(mplx_no):
                memeplexes[i, j] = sorted_fitness[i+(mplx_no*j)]
        return memeplexes

    def local_search(self, frogs, memeplex):
        """Performs the local search for a memeplex.

        Arguments:
            frogs {numpy.ndarray} -- All the frogs
            memeplex {numpy.ndarray} -- One memeplex

        Returns:
            numpy.ndarray -- The updated frogs, same dimensions
        """

        # Select worst, best, greatest frogs
        frog_w = frogs[int(memeplex[-1])]
        frog_b = frogs[int(memeplex[0])]
        frog_g = frogs[0]
        # Move worst wrt best frog
        frog_w_new = frog_w + (np.random.rand() * (frog_b - frog_w))
        # If change not better, move worst wrt greatest frog
        if self.opt_func(frog_w_new) > self.opt_func(frog_w):
            frog_w_new = frog_w + (np.random.rand() * (frog_g - frog_w))
        # If change not better, random new worst frog
        if self.opt_func(frog_w_new) > self.opt_func(frog_w):
            frog_w_new = self.gen_frogs(1)[0]
        # Replace worst frog
        frogs[int(memeplex[-1])] = frog_w_new
        return frogs

    @staticmethod
    def shuffle_memeplexes(memeplexes):
        """Shuffles the memeplexes without sorting them.

        Arguments:
            memeplexes {numpy.ndarray} -- The memeplexes

        Returns:
            numpy.ndarray -- A shuffled memeplex, unsorted, same dimensions
        """

        # Flatten the array
        temp = memeplexes.flatten()
        # Shuffle the array
        np.random.shuffle(temp)
        # Reshape
        temp = temp.reshape((memeplexes.shape[0], memeplexes.shape[1]))
        return temp

# Entry point, con todos los argumentos que se requieren. Tal vez mover para acá el numero de ranas y de mplx
    def sfla(self, start_pos, target_pos, obstacles, mplx_iters=10, solun_iters=30):
        """Performs the Shuffled Leaping Frog Algorithm.

        Arguments:
            opt_func {function} -- The function to optimize.

        Keyword Arguments:
            frogs {int} -- The number of frogs to use (default: {30})
            sigma {int/float} -- Sigma for the gaussian normal distribution to create the frogs (default: {1})
            center {int/float} -- 
            mplx_no {int} -- Number of memeplexes, when divides frog number should return an integer otherwise frogs will be skipped (default: {6})
            mplx_iters {int} -- Number of times a single memeplex will be iterated before shuffling (default: {10})
            solun_iters {int} -- Number of times the memeplexes will be shuffled (default: {50})

        Returns:
            tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) -- [description]
        """
        self.start = start_pos
        self.target = target_pos
        self.obstacles = obstacles
        # Generate frogs around the given position
        frogs = self.gen_frogs(self.frogs_no)
        # Arrange frogs and sort into memeplexes
        memeplexes = self.sort_frogs(frogs, self.mplx_no)
        # Best solution as greatest frog
        best_solun = frogs[int(memeplexes[0, 0])]
        # For the number of iterations
        for i in range(solun_iters):
            # Shuffle memeplexes
            memeplexes = self.shuffle_memeplexes(memeplexes)
            # For each memeplex
            for mplx_idx, memeplex in enumerate(memeplexes):
                # For number of memeplex iterations
                for j in range(mplx_iters):
                    # Perform local search
                    frogs = self.local_search(frogs, memeplex)
                # Rearrange memeplexes
                memeplexes = self.sort_frogs(frogs, self.mplx_no)
                # Check and select new best frog as the greatest frog
                new_best_solun = frogs[int(memeplexes[0, 0])]
                if self.opt_func(new_best_solun) < self.opt_func(best_solun):
                    best_solun = new_best_solun
        return best_solun, frogs, memeplexes.astype(int)
