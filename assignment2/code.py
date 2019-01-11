import math
import os
import warnings
from functools import partial
from multiprocessing import cpu_count, Pool

import pandas as pd
import requests
from geopy.distance import distance
from geopy.geocoders import Nominatim
from tqdm import tqdm_notebook as tqdm

USER_AGENT = "stat159-group-assignment"
MAX_LATITUDE = 90


####################################################################################
# UTILITY FUNCTIONS
####################################################################################

def find_zero(func, guess=1.0):
    """
    Use Newton's method to solve for an input of the function such that the
    function will return zero.

    Reference: Part of this function is adapted from https://cs61a.org/extra/extra01/

    :param func:    A function which takes a float number as input
    :param guess:   The initial guess of the input
    :return:        An input such that the function returns zero
    """

    def improve(update, close, _guess=1.0, _max_updates=100):
        """Iteratively improve guess with update until close(guess) is true."""
        k = 0
        while not close(_guess) and k < _max_updates:
            _guess = update(_guess)
            k = k + 1
        return _guess

    def approx_eq(x, y, tolerance=1e-5):
        """Whether x is within tolerance of y."""
        return abs(x - y) < tolerance

    def near_zero(x):
        """Whether x is near zero."""
        return approx_eq(func(x), 0)

    def newton_update(f, df):
        """Return an update function for f with derivative df."""
        return lambda x: x - f(x) / max(df(x), 1e-16)

    def differentiate(f, delta=1e-5):
        """Approximately differentiate a single-argument function."""
        return lambda x: (f(x + delta) - f(x - delta)) / (2 * delta)

    return improve(newton_update(func, differentiate(func)), near_zero, guess)


def parallel_list_map(lst, func, num_partitions=100, **kwargs):
    """
    Multi-threading helper to apply a function on a list. This function
    will return the same result as calling func(lst, **kwargs) directly.

    The function must take a list as input (may have other arguments) and
    return a list as its output.

    :param lst             The input list
    :param func            The function to be applied on the list
    :param num_partitions  Number of threads
    :return:               The same output list as returned by func(lst)
    """
    # Split the list based on number of partitions
    lst_split = [lst[i::num_partitions] for i in range(num_partitions)]
    # Create a thread pool
    pool = Pool(cpu_count())
    # Run the function and concatenate the result
    lst = sum(pool.map(partial(func, **kwargs), lst_split), [])
    # Clean up
    pool.close()
    pool.join()
    return lst


def download_and_cache_file(name, url, force_download=False):
    """
    Download and cache a file of given NAME from the given URL. The file
    will be cached in the data folder.
    A progress bar will be displayed if run within Jupyter notebook.

    :param name:            Name of the file
    :param url:             Download url of the file
    :param force_download:  Whether ignore local cache and re-download the file
    :return:                Path to the local file
    """
    cache_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', name)
    if not os.path.isfile(cache_file) or force_download:
        print("Downloading {} ...".format(os.path.basename(cache_file)))
        r = requests.get(url, stream=True)
        total_size, block_size = int(r.headers.get('content-length', 0)), 1024
        wrote = 0
        with open(cache_file, 'wb') as f:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=ImportWarning)
                warnings.simplefilter(action='ignore', category=DeprecationWarning)
                for data in tqdm(r.iter_content(block_size),
                                 total=math.ceil(total_size // block_size), unit_scale=True):
                    wrote = wrote + len(data)
                    f.write(data)
        print("File {} downloaded from {}".format(os.path.basename(cache_file), url))
    return cache_file


####################################################################################
# FUNCTIONS FOR GENERATING THE GRID
####################################################################################

def get_county_rectangle_border(county):
    """
    Get the rectangle border of the given county using Nominatim.

    :param county:   Name of the county
    :return:         Coordinates of the south-west and north-east corners.
                     Returned as a list of two tuples, where each tuple is
                     in the format of (latitude, longitude).
    """
    # Location of the county
    geo_locator = Nominatim(user_agent=USER_AGENT)
    location = geo_locator.geocode({"county": county})

    # Get the bounds of the county from the result obtained from Nominatim
    # (Nominatim gives the south-west and north-east corners)
    bounds = list(map(float, location.raw['boundingbox']))

    return [(bounds[0], bounds[2]), (bounds[1], bounds[3])]


def get_county_polygon_border(county):
    """
    Get the polygon border of the given county using Nominatim.

    :param county:    Name of the county
    :return:          Coordinates of all turning points in polygon
                      border of the county in counter-clock-wise order.
                      Returned as a list of tuples, where each tuple is
                      in the format of (latitude, longitude).
    """
    # Location of the county
    geo_locator = Nominatim(user_agent=USER_AGENT, timeout=5)
    geo_locator.structured_query_params.add("polygon_geojson")
    location = geo_locator.geocode({"county": county, "polygon_geojson": 1, "format": "json"})

    # Turning points on the polygon border is given in (lon, lat) format,
    # we need to re-order it into (lat, lon)
    return list(map(lambda x: (x[1], x[0]), location.raw['geojson']['coordinates'][0]))


def generate_grid_within_rectangle_bounds(county, start_point=None, spacing=5):
    """
    Generate a grid within the rectangle bound of the given county

    :param county:        Name of the county
    :param start_point:   A starting top-right-corner coordinate (lat, lon) that
                          need to be included
    :param spacing:       Spacing between each point within the grid
    :return:              Coordinates within the bound as a list of tuples where
                          each tuple is in the format of (latitude, longitude).
    """
    bounds = get_county_rectangle_border(county)
    lat_range = [bounds[0][0], bounds[1][0]] + ([start_point[0]] if start_point else [])
    lon_range = [bounds[0][1], bounds[1][1]] + ([start_point[1]] if start_point else [])

    grid = []

    lat = start_point[0] if start_point else max(lat_range)
    while lat >= min(lat_range):
        lon = start_point[1] if start_point else min(lon_range)
        while lon < max(lon_range):
            # Add the coordinate
            grid.append((lat, lon))
            # Update longitude by using Newton's method to find the first longitude which is 5 miles away
            lon += find_zero(lambda delta_lon:
                             distance((lat, lon), (lat, lon + delta_lon)).miles - spacing, guess=0.09)

        # Update latitude by using Newton's method to find the first latitude which is 5 miles away
        lat -= find_zero(lambda delta_lat:
                         distance((lat, lon), (lat - delta_lat, lon)).miles - spacing, guess=0.07)

    return grid


def check_coordinate_within_county(coordinate, county):
    """
    Check if a given coordinate is within the given county

    :param coordinate: Coordinate represented as a tuple (latitude, longitude)
    :param county:     Name of the county
    :return:           (boolean) whether the point is within the given county
    """

    # Reverse search for the address of the given geo-coordinate
    geo_locator = Nominatim(user_agent=USER_AGENT, timeout=5)
    location = geo_locator.reverse("{}, {}".format(*coordinate), addressdetails=True)

    # Return if the county is the same as the given county
    return ('county' in location.raw['address'] and
            location.raw['address']['county'] == county)


def filter_grid(grid, county):
    """
    Filter the grid base on whether it is within the given county

    :param grid:    Coordinates within the grid as a list of tuples
    :param county:  Name of the county
    :return:        A filtered list of coordinates where all points outside the county
                    are removed
    """
    return list(filter(lambda coordinate:
                       check_coordinate_within_county(coordinate, county), grid))


def filter_grid_parallel(grid, county):
    """
    Filter the grid base on whether it is within the given county

    :param grid:    Coordinates within the grid as a list of tuples
    :param county:  Name of the county
    :return:        A filtered list of coordinates where all points outside the county
                    are removed
    """
    return parallel_list_map(grid, filter_grid, county=county)


####################################################################################
# FUNCTIONS FOR PROCESSING WEATHER STATIONS
####################################################################################

def load_stations(url="https://osf.io/ev7y2/download"):
    """
    Load all weather stations from the given FILE 

    :param url:             Download URL of the csv file
    :return:                A list of stations
    """
    # Download file if not exist
    cache_file = download_and_cache_file("stations_ca.csv", url)

    # Load data from csv file
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        stations_data = pd.read_csv(cache_file, index_col=0, low_memory=False)

    return stations_data.to_dict(orient='records')


def filter_stations(stations, grid, max_distance=10):
    """
    Filter stations to only keep those within MAX_DISTANCE from the any grid
    point of the given COUNTY

    :param stations:      A list of dictionaries which each represents a station
    :param grid:          Coordinates within the grid as a list of tuples
    :param max_distance:  Maximum distance for the station to be considered
    :return:              A list of coordinates that is within MAX_DISTANCE
                          from the border of the COUNTY
    """

    def is_valid_station(station):
        for grid_point in grid:
            if distance(grid_point, station) <= max_distance:
                return True
        return False

    return list(filter(lambda station: is_valid_station((station['LATITUDE'], station['LONGITUDE'])), stations))


def filter_stations_parallel(stations, grid, max_distance=10):
    """A multi-threading version of filter_stations """
    return parallel_list_map(stations, filter_stations, grid=grid, max_distance=max_distance)


def filter_stations_ec(stations, county, max_distance=5):
    """
    [EXTRA CREDIT]
    Filter stations to only keep those within MAX_DISTANCE from the polygon
    border of the given COUNTY

    :param stations:      A list of dictionaries which each represents a station
    :param county:        Name of the county
    :param max_distance:  Maximum distance for the station to be considered
    :return:              A list of coordinates that is within MAX_DISTANCE
                          from the border of the COUNTY
    """

    def is_in_boundary(station, polygon_border):
        """
        Check if a station coordinate is within the polygon border

        :param station:         The coordinate of the station (lat, lon)
        :param polygon_border:  Polygon border represented as a list of coordinates (lat, lon)
        :return:                Whether the polygon contains the coordinate
        """

        def on_segment(p, q, r):
            return (max(p[0], r[0]) >= q[0] >= min(p[0], r[0])
                    and max(p[1], r[1]) >= q[1] >= min(p[1], r[1]))

        def get_orientation(p, q, r):
            indicator = ((q[1] - p[1]) * (r[0] - q[0]) -
                         (q[0] - p[0]) * (r[1] - q[1]))
            if indicator == 0:
                return 0
            return 1 if indicator > 0 else 2

        def is_intersect(p1, q1, p2, q2):
            o1, o2, o3, o4 = (get_orientation(p1, q1, p2),
                              get_orientation(p1, q1, q2),
                              get_orientation(p2, q2, p1),
                              get_orientation(p2, q2, q1))
            return ((o1 != o2 and o3 != o4)
                    or (o1 == 0 and on_segment(p1, p2, q1))
                    or (o2 == 0 and on_segment(p1, q2, q1))
                    or (o3 == 0 and on_segment(p2, q1, q2))
                    or (o4 == 0 and on_segment(p2, q1, q2)))

        n = len(polygon_border)
        count = 0
        extreme = (MAX_LATITUDE, station[1])

        for i in range(n):
            next_i = (i + 1) % n
            if is_intersect(polygon_border[i], polygon_border[next_i], station, extreme):
                if get_orientation(polygon_border[i], station, polygon_border[next_i]) == 0:
                    return on_segment(polygon_border[i], station, polygon_border[next_i])
                count += 1
        return count % 2 == 1

    def is_within_dist(station, polygon_border, rectangle_border):
        """
        Check if a station coordinate is within MAX_DISTANCE of polygon border

        :param station:          The coordinate of the station (lat, lon)
        :param polygon_border:   Polygon border represented as a list of coordinates (lat, lon)
        :param rectangle_border: Rectangle border represented as a list of two coordinates:
                                 the south-west and north-east corners of the county
        :return:                 Whether the coordinate is MAX_DISTANCE away from the border
        """

        def can_intersect(point, r, line):
            """
            Return whether a circle of radius R, centered at POINT, can intersect the LINE
            :param point:   Coordinate of a point given as (lat, lon)
            :param r:       Radius of the circle
            :param line:    A line segment given as [(lat1, lon1), (lat2, lon2)]
            :return:        Whether the point is R miles away from the line
            """

            # If the distance to the two endpoints is less than R, return True
            if distance(point, line[0]).miles <= r or distance(point, line[1]).miles <= r:
                return True

            # Find the perpendicular distance from the point to the line
            lat, lon = point
            lat1, lon1 = line[0]
            lat2, lon2 = line[1]

            # Find the point (x,y) on the line such that it is the intersection of the
            # perpendicular line from the point (lat, lon)
            if lat1 == lat2:
                x, y = lat1, lon
            elif lon1 == lon2:
                x, y = lat, lon1
            else:
                slope = (lon2 - lon1) / (lat2 - lat1)
                x = (lat / slope - lon1 + lon + slope * lat1) / (slope + 1 / slope)
                y = lon1 + slope * (x - lat1)

            # Only consider the situation where (x,y) is within the given line segment
            return (min(lat1, lat2) <= x <= max(lat1, lat2)
                    and min(lon1, lon2) <= y <= max(lon1, lon2)
                    and distance((x, y), point).miles <= r)

        # First filter out points that are too far away
        lat_range = [rectangle_border[0][0], rectangle_border[1][0]]
        lon_range = [rectangle_border[0][1], rectangle_border[1][1]]
        if not (min(lat_range) - max_distance * 0.02 <= station[0] <= max(lat_range) + max_distance * 0.02
                and min(lon_range) - max_distance * 0.02 <= station[1] <= max(lon_range) + max_distance * 0.02):
            return False

        # Now use the polygon border to check if the point is valid
        for border_line in zip(polygon_border, polygon_border[1:]):
            if can_intersect(station, max_distance, border_line):
                return True
        return False

    # Combine two conditions above
    def is_valid_station(station, polygon_border, rectangle_border):
        """Whether the station is either in boundary or within 5 miles"""
        return is_in_boundary(station, polygon_border) or is_within_dist(station, polygon_border, rectangle_border)

    county_polygon_border = get_county_polygon_border(county)[::-1]
    county_rectangle_border = get_county_rectangle_border(county)

    return list(filter(lambda station: is_valid_station((station['LATITUDE'], station['LONGITUDE']),
                                                        county_polygon_border, county_rectangle_border), stations))


def filter_stations_ec_parallel(stations, county, max_distance=5):
    """A multi-threading version of filter_stations_ec """
    return parallel_list_map(stations, filter_stations_ec, num_partitions=cpu_count() * 2,
                             county=county, max_distance=max_distance)


def convert_stations_to_coordinates(stations):
    """
    Convert a list of station objects to a list of (lat, lon) tuple
    :param stations:  A list of station objects
    :return:          A list of (lat, lon) tuple
    """
    return list(map(lambda s: (s['LATITUDE'], s['LONGITUDE']), stations))
