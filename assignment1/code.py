import numpy as np
from geopy.distance import distance

# Lower limit of the distance to avoid division by zero
MIN_DISTANCE = 1e-6


def generate_random_grid():
    """
    Generate a random grid centered at the origin

    :return:     (List of tuples) Random coordinates
    """
    # Generate the grid as N * 2 numpy matrix
    lat_grid, lon_grid = np.meshgrid(np.arange(-0.5, 0.5, 0.1),
                                     np.arange(-0.5, 0.5, 0.1))
    grid = np.concatenate((lat_grid.ravel().reshape(-1, 1), lon_grid.ravel().reshape(-1, 1)), axis=1)

    # Convert numpy matrix to list of tuples
    return list(zip(grid[:, 0], grid[:, 1]))


def generate_random_stations(num=10):
    """
    Generate random coordinates centered at the origin

    :param num:  Number of coordinates to generate
    :return:     (List of tuples) Random coordinates
    """
    # Generate the station coordinates as K * 2 numpy matrix
    stations = (np.random.rand(num, 2) - 0.5) * 1.5

    # Convert numpy matrix to list of tuples
    return list(zip(stations[:, 0], stations[:, 1]))


def compute_average_inverse_distance(grid, stations):
    """
    Compute the average inverse distance from the given coordinate to the grid

    :param grid:     (List of tuples) The latitude and longitude of points that belong to the grid
    :param stations: (List of tuples) The latitude and longitude of the stations

    :return: (List) Average inverse distances in miles
    """
    # Convert list of tuples to N x 2 numpy matrix
    grid, stations = np.array(grid), np.array(stations)

    def compute_single_coordinate(coordinate):
        """
        Compute the mean inverse-distance for a single station
        :param coordinate: (Tuples) The latitude and longitude of a station
        """
        return np.average(np.reciprocal(
            np.apply_along_axis(
                lambda row: max(distance(tuple(row), tuple(coordinate)).miles, MIN_DISTANCE),
                axis=1, arr=grid)))

    # Compute distance for each station
    distances = np.apply_along_axis(compute_single_coordinate, 1, stations)
    return distances.ravel().tolist()
