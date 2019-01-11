from geopy.geocoders import GoogleV3

GOOGLE_MAP_API_KEY = "AIzaSyBTHDv-ei4m_2CqFFHEBwGx_rg27q1L3aA"


def check_coordinate_within_zipcodes(coordinate, zipcodes):
    """
    Check if a given coordinate is within the given zipcodes 

    :param coordinate: Coordinate represented as a tuple (lat, lon)
    :param zipcodes:   Reference zip codes
    :return:           (boolean) whether the point is within the given zip codes
    """

    # Reverse search for the address of the given geo-coordinate with GoogleV3
    geo_locator = GoogleV3(api_key=GOOGLE_MAP_API_KEY)
    location = geo_locator.reverse("{}, {}".format(*coordinate), exactly_one=False)

    addresses = sum(map(lambda loc: loc.raw['address_components'], location), [])
    zip_code = (list(filter(lambda address: 'postal_code' in address['types'], addresses))[0]['short_name'])
    return int(zip_code) in zipcodes


def split_grid(grid, east_zipcodes):
    """
    Split the grid base on whether it is within the given county

    :param grid:    Coordinates within the grid as a list of tuples
    :param east_zipcodes:  zipcodes of the east side
    :return:        two list of coordinates where one contains the points
                    on the West Alameda and one contains those on the East Alameda.
    """
    east_grid, west_grid = [], []
    for coordinate in grid:
        if check_coordinate_within_zipcodes(coordinate, east_zipcodes):
            east_grid.append(coordinate)
        else:
            west_grid.append(coordinate)
    return east_grid, west_grid
