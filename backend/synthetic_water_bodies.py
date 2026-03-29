"""
TrophicTrace — Synthetic Water Body Generator
Creates plausible PFAS monitoring locations across all 50 US states using
real water body names, geographic coordinates, and realistic contamination profiles.

These fill the map where we lack real EPA WQP monitoring data.
"""

import numpy as np
import math

# ============================================================
# Real US water bodies — name, lat, lng, state, typical water type
# Sourced from NHDPlus, USGS GNIS, and EPA 303(d) listed waters
# ============================================================
WATER_BODIES = [
    # ---------- NORTHEAST ----------
    # Maine
    {"name": "Androscoggin River — Lewiston", "lat": 44.10, "lng": -70.22, "state": "ME", "type": "river", "urban": 35},
    {"name": "Kennebec River — Augusta", "lat": 44.31, "lng": -69.77, "state": "ME", "type": "river", "urban": 25},
    {"name": "Penobscot River — Bangor", "lat": 44.80, "lng": -68.77, "state": "ME", "type": "river", "urban": 30},
    {"name": "Sebago Lake — Windham", "lat": 43.84, "lng": -70.55, "state": "ME", "type": "lake", "urban": 15},
    {"name": "Presumpscot River — Westbrook", "lat": 43.69, "lng": -70.37, "state": "ME", "type": "river", "urban": 40},
    # New Hampshire
    {"name": "Merrimack River — Concord", "lat": 43.21, "lng": -71.54, "state": "NH", "type": "river", "urban": 40},
    {"name": "Piscataqua River — Portsmouth", "lat": 43.07, "lng": -70.76, "state": "NH", "type": "river", "urban": 50},
    {"name": "Lake Winnipesaukee — Laconia", "lat": 43.57, "lng": -71.36, "state": "NH", "type": "lake", "urban": 20},
    {"name": "Connecticut River — Hanover", "lat": 43.70, "lng": -72.29, "state": "NH", "type": "river", "urban": 15},
    # Vermont
    {"name": "Lake Champlain — Burlington", "lat": 44.48, "lng": -73.21, "state": "VT", "type": "lake", "urban": 35},
    {"name": "Otter Creek — Middlebury", "lat": 44.01, "lng": -73.17, "state": "VT", "type": "river", "urban": 15},
    {"name": "Winooski River — Montpelier", "lat": 44.26, "lng": -72.58, "state": "VT", "type": "river", "urban": 25},
    {"name": "Walloomsac River — Bennington", "lat": 42.88, "lng": -73.20, "state": "VT", "type": "river", "urban": 30},
    # Massachusetts
    {"name": "Charles River — Boston", "lat": 42.37, "lng": -71.07, "state": "MA", "type": "river", "urban": 85},
    {"name": "Merrimack River — Lowell", "lat": 42.64, "lng": -71.31, "state": "MA", "type": "river", "urban": 65},
    {"name": "Connecticut River — Springfield", "lat": 42.10, "lng": -72.59, "state": "MA", "type": "river", "urban": 55},
    {"name": "Taunton River — Taunton", "lat": 41.90, "lng": -71.09, "state": "MA", "type": "river", "urban": 45},
    {"name": "Quabbin Reservoir — Belchertown", "lat": 42.38, "lng": -72.33, "state": "MA", "type": "lake", "urban": 10},
    {"name": "Neponset River — Milton", "lat": 42.25, "lng": -71.07, "state": "MA", "type": "river", "urban": 70},
    {"name": "Cape Cod Bay — Barnstable", "lat": 41.70, "lng": -70.30, "state": "MA", "type": "estuary", "urban": 40},
    # Rhode Island
    {"name": "Narragansett Bay — Providence", "lat": 41.73, "lng": -71.34, "state": "RI", "type": "estuary", "urban": 75},
    {"name": "Blackstone River — Pawtucket", "lat": 41.88, "lng": -71.38, "state": "RI", "type": "river", "urban": 70},
    {"name": "Pawcatuck River — Westerly", "lat": 41.38, "lng": -71.83, "state": "RI", "type": "river", "urban": 30},
    # Connecticut
    {"name": "Connecticut River — Hartford", "lat": 41.76, "lng": -72.68, "state": "CT", "type": "river", "urban": 65},
    {"name": "Housatonic River — New Milford", "lat": 41.58, "lng": -73.41, "state": "CT", "type": "river", "urban": 35},
    {"name": "Thames River — New London", "lat": 41.35, "lng": -72.10, "state": "CT", "type": "river", "urban": 45},
    {"name": "Quinnipiac River — Meriden", "lat": 41.54, "lng": -72.80, "state": "CT", "type": "river", "urban": 55},
    {"name": "Naugatuck River — Waterbury", "lat": 41.56, "lng": -73.04, "state": "CT", "type": "river", "urban": 50},
    # New York
    {"name": "Hudson River — Albany", "lat": 42.65, "lng": -73.76, "state": "NY", "type": "river", "urban": 55},
    {"name": "Hudson River — Poughkeepsie", "lat": 41.70, "lng": -73.93, "state": "NY", "type": "river", "urban": 45},
    {"name": "Mohawk River — Schenectady", "lat": 42.81, "lng": -73.94, "state": "NY", "type": "river", "urban": 50},
    {"name": "Finger Lakes — Seneca Lake", "lat": 42.66, "lng": -76.92, "state": "NY", "type": "lake", "urban": 15},
    {"name": "Lake Ontario — Rochester", "lat": 43.26, "lng": -77.62, "state": "NY", "type": "lake", "urban": 50},
    {"name": "Niagara River — Buffalo", "lat": 42.88, "lng": -78.88, "state": "NY", "type": "river", "urban": 60},
    {"name": "Genesee River — Rochester", "lat": 43.16, "lng": -77.61, "state": "NY", "type": "river", "urban": 55},
    {"name": "Susquehanna River — Binghamton", "lat": 42.10, "lng": -75.91, "state": "NY", "type": "river", "urban": 40},
    {"name": "Long Island Sound — Oyster Bay", "lat": 40.87, "lng": -73.53, "state": "NY", "type": "estuary", "urban": 75},
    {"name": "Jamaica Bay — Queens", "lat": 40.61, "lng": -73.82, "state": "NY", "type": "estuary", "urban": 90},
    {"name": "Hoosic River — Hoosick Falls", "lat": 42.90, "lng": -73.35, "state": "NY", "type": "river", "urban": 25},
    # New Jersey
    {"name": "Delaware River — Trenton", "lat": 40.22, "lng": -74.77, "state": "NJ", "type": "river", "urban": 60},
    {"name": "Passaic River — Newark", "lat": 40.74, "lng": -74.17, "state": "NJ", "type": "river", "urban": 85},
    {"name": "Raritan River — New Brunswick", "lat": 40.49, "lng": -74.45, "state": "NJ", "type": "river", "urban": 65},
    {"name": "Hackensack River — Hackensack", "lat": 40.89, "lng": -74.04, "state": "NJ", "type": "river", "urban": 80},
    {"name": "Barnegat Bay — Toms River", "lat": 39.95, "lng": -74.13, "state": "NJ", "type": "estuary", "urban": 50},
    # Pennsylvania
    {"name": "Schuylkill River — Philadelphia", "lat": 39.97, "lng": -75.19, "state": "PA", "type": "river", "urban": 80},
    {"name": "Delaware River — Philadelphia", "lat": 39.95, "lng": -75.14, "state": "PA", "type": "river", "urban": 85},
    {"name": "Susquehanna River — Harrisburg", "lat": 40.26, "lng": -76.88, "state": "PA", "type": "river", "urban": 50},
    {"name": "Lehigh River — Allentown", "lat": 40.60, "lng": -75.47, "state": "PA", "type": "river", "urban": 55},
    {"name": "Allegheny River — Pittsburgh", "lat": 40.45, "lng": -79.98, "state": "PA", "type": "river", "urban": 65},
    {"name": "Monongahela River — Pittsburgh", "lat": 40.43, "lng": -79.99, "state": "PA", "type": "river", "urban": 60},
    {"name": "Neshaminy Creek — Warminster", "lat": 40.20, "lng": -75.10, "state": "PA", "type": "river", "urban": 55},
    {"name": "Lake Erie — Erie", "lat": 42.13, "lng": -80.08, "state": "PA", "type": "lake", "urban": 45},

    # ---------- SOUTHEAST ----------
    # Delaware / Maryland
    {"name": "Chesapeake Bay — Annapolis", "lat": 38.97, "lng": -76.49, "state": "MD", "type": "estuary", "urban": 55},
    {"name": "Patuxent River — Laurel", "lat": 39.10, "lng": -76.85, "state": "MD", "type": "river", "urban": 50},
    {"name": "Potomac River — Bethesda", "lat": 38.98, "lng": -77.10, "state": "MD", "type": "river", "urban": 70},
    {"name": "Back River — Baltimore", "lat": 39.25, "lng": -76.48, "state": "MD", "type": "river", "urban": 75},
    {"name": "Christina River — Wilmington", "lat": 39.73, "lng": -75.56, "state": "DE", "type": "river", "urban": 60},
    {"name": "Indian River Bay — Rehoboth", "lat": 38.60, "lng": -75.07, "state": "DE", "type": "estuary", "urban": 35},
    # Virginia
    {"name": "James River — Richmond", "lat": 37.53, "lng": -77.43, "state": "VA", "type": "river", "urban": 55},
    {"name": "Potomac River — Alexandria", "lat": 38.80, "lng": -77.05, "state": "VA", "type": "river", "urban": 75},
    {"name": "Shenandoah River — Front Royal", "lat": 38.92, "lng": -78.19, "state": "VA", "type": "river", "urban": 15},
    {"name": "York River — Yorktown", "lat": 37.24, "lng": -76.51, "state": "VA", "type": "river", "urban": 25},
    {"name": "Dan River — Danville", "lat": 36.59, "lng": -79.39, "state": "VA", "type": "river", "urban": 35},
    {"name": "Elizabeth River — Norfolk", "lat": 36.84, "lng": -76.29, "state": "VA", "type": "river", "urban": 70},
    # West Virginia
    {"name": "Kanawha River — Charleston", "lat": 38.35, "lng": -81.63, "state": "WV", "type": "river", "urban": 40},
    {"name": "Ohio River — Parkersburg", "lat": 39.27, "lng": -81.56, "state": "WV", "type": "river", "urban": 35},
    {"name": "Monongahela River — Morgantown", "lat": 39.63, "lng": -79.96, "state": "WV", "type": "river", "urban": 35},
    # North Carolina
    {"name": "Cape Fear River — Fayetteville", "lat": 35.05, "lng": -78.88, "state": "NC", "type": "river", "urban": 45},
    {"name": "Cape Fear River — Wilmington", "lat": 34.23, "lng": -77.94, "state": "NC", "type": "river", "urban": 50},
    {"name": "Neuse River — Raleigh", "lat": 35.78, "lng": -78.64, "state": "NC", "type": "river", "urban": 55},
    {"name": "Yadkin River — Winston-Salem", "lat": 36.10, "lng": -80.26, "state": "NC", "type": "river", "urban": 45},
    {"name": "Catawba River — Hickory", "lat": 35.73, "lng": -81.34, "state": "NC", "type": "river", "urban": 30},
    {"name": "Haw River — Burlington", "lat": 36.10, "lng": -79.43, "state": "NC", "type": "river", "urban": 35},
    {"name": "French Broad River — Asheville", "lat": 35.60, "lng": -82.55, "state": "NC", "type": "river", "urban": 35},
    {"name": "Tar River — Greenville", "lat": 35.61, "lng": -77.37, "state": "NC", "type": "river", "urban": 30},
    # South Carolina
    {"name": "Savannah River — Augusta", "lat": 33.47, "lng": -81.97, "state": "SC", "type": "river", "urban": 40},
    {"name": "Congaree River — Columbia", "lat": 33.98, "lng": -81.04, "state": "SC", "type": "river", "urban": 50},
    {"name": "Cooper River — Charleston", "lat": 32.84, "lng": -79.96, "state": "SC", "type": "river", "urban": 50},
    {"name": "Pee Dee River — Florence", "lat": 34.20, "lng": -79.76, "state": "SC", "type": "river", "urban": 30},
    # Georgia
    {"name": "Chattahoochee River — Atlanta", "lat": 33.84, "lng": -84.45, "state": "GA", "type": "river", "urban": 75},
    {"name": "Savannah River — Savannah", "lat": 32.08, "lng": -81.10, "state": "GA", "type": "river", "urban": 45},
    {"name": "Oconee River — Athens", "lat": 33.95, "lng": -83.38, "state": "GA", "type": "river", "urban": 40},
    {"name": "Altamaha River — Jesup", "lat": 31.61, "lng": -81.88, "state": "GA", "type": "river", "urban": 15},
    {"name": "Ocmulgee River — Macon", "lat": 32.84, "lng": -83.63, "state": "GA", "type": "river", "urban": 40},
    # Florida
    {"name": "St. Johns River — Jacksonville", "lat": 30.32, "lng": -81.66, "state": "FL", "type": "river", "urban": 60},
    {"name": "Hillsborough River — Tampa", "lat": 28.01, "lng": -82.46, "state": "FL", "type": "river", "urban": 65},
    {"name": "Caloosahatchee River — Fort Myers", "lat": 26.66, "lng": -81.87, "state": "FL", "type": "river", "urban": 45},
    {"name": "Miami Canal — Miami", "lat": 25.79, "lng": -80.24, "state": "FL", "type": "river", "urban": 85},
    {"name": "Lake Okeechobee — Belle Glade", "lat": 26.95, "lng": -80.80, "state": "FL", "type": "lake", "urban": 20},
    {"name": "Apalachicola River — Apalachicola", "lat": 29.73, "lng": -85.03, "state": "FL", "type": "river", "urban": 10},
    {"name": "Escambia Bay — Pensacola", "lat": 30.45, "lng": -87.19, "state": "FL", "type": "estuary", "urban": 50},
    {"name": "Indian River Lagoon — Melbourne", "lat": 28.08, "lng": -80.60, "state": "FL", "type": "estuary", "urban": 45},
    {"name": "Tampa Bay — St. Petersburg", "lat": 27.77, "lng": -82.63, "state": "FL", "type": "estuary", "urban": 70},

    # ---------- MIDWEST ----------
    # Ohio
    {"name": "Cuyahoga River — Cleveland", "lat": 41.50, "lng": -81.69, "state": "OH", "type": "river", "urban": 75},
    {"name": "Great Miami River — Dayton", "lat": 39.76, "lng": -84.19, "state": "OH", "type": "river", "urban": 55},
    {"name": "Scioto River — Columbus", "lat": 39.96, "lng": -83.00, "state": "OH", "type": "river", "urban": 60},
    {"name": "Maumee River — Toledo", "lat": 41.65, "lng": -83.54, "state": "OH", "type": "river", "urban": 55},
    {"name": "Tuscarawas River — Canton", "lat": 40.80, "lng": -81.38, "state": "OH", "type": "river", "urban": 40},
    {"name": "Lake Erie — Sandusky", "lat": 41.45, "lng": -82.71, "state": "OH", "type": "lake", "urban": 35},
    # Michigan
    {"name": "Grand River — Grand Rapids", "lat": 42.96, "lng": -85.67, "state": "MI", "type": "river", "urban": 55},
    {"name": "Kalamazoo River — Kalamazoo", "lat": 42.29, "lng": -85.59, "state": "MI", "type": "river", "urban": 45},
    {"name": "Huron River — Ann Arbor", "lat": 42.28, "lng": -83.75, "state": "MI", "type": "river", "urban": 50},
    {"name": "Saginaw River — Saginaw", "lat": 43.42, "lng": -83.95, "state": "MI", "type": "river", "urban": 45},
    {"name": "Rouge River — Dearborn", "lat": 42.31, "lng": -83.21, "state": "MI", "type": "river", "urban": 80},
    {"name": "Au Sable River — Oscoda", "lat": 44.42, "lng": -83.33, "state": "MI", "type": "river", "urban": 10},
    {"name": "Lake Michigan — Traverse City", "lat": 44.76, "lng": -85.62, "state": "MI", "type": "lake", "urban": 20},
    {"name": "Clark Fork — Parchment", "lat": 42.33, "lng": -85.57, "state": "MI", "type": "river", "urban": 30},
    # Indiana
    {"name": "White River — Indianapolis", "lat": 39.77, "lng": -86.16, "state": "IN", "type": "river", "urban": 65},
    {"name": "Wabash River — Lafayette", "lat": 40.42, "lng": -86.90, "state": "IN", "type": "river", "urban": 40},
    {"name": "St. Joseph River — South Bend", "lat": 41.68, "lng": -86.25, "state": "IN", "type": "river", "urban": 45},
    {"name": "White River — Muncie", "lat": 40.19, "lng": -85.39, "state": "IN", "type": "river", "urban": 35},
    # Illinois
    {"name": "Chicago River — Chicago", "lat": 41.89, "lng": -87.63, "state": "IL", "type": "river", "urban": 90},
    {"name": "Illinois River — Peoria", "lat": 40.69, "lng": -89.59, "state": "IL", "type": "river", "urban": 40},
    {"name": "Des Plaines River — Joliet", "lat": 41.53, "lng": -88.08, "state": "IL", "type": "river", "urban": 60},
    {"name": "Sangamon River — Springfield", "lat": 39.80, "lng": -89.64, "state": "IL", "type": "river", "urban": 40},
    {"name": "Fox River — Elgin", "lat": 42.04, "lng": -88.28, "state": "IL", "type": "river", "urban": 55},
    {"name": "Rock River — Rockford", "lat": 42.27, "lng": -89.09, "state": "IL", "type": "river", "urban": 45},
    # Wisconsin
    {"name": "Milwaukee River — Milwaukee", "lat": 43.04, "lng": -87.91, "state": "WI", "type": "river", "urban": 65},
    {"name": "Wisconsin River — Wisconsin Dells", "lat": 43.63, "lng": -89.77, "state": "WI", "type": "river", "urban": 20},
    {"name": "Fox River — Green Bay", "lat": 44.51, "lng": -88.02, "state": "WI", "type": "river", "urban": 45},
    {"name": "Rock River — Janesville", "lat": 42.68, "lng": -89.02, "state": "WI", "type": "river", "urban": 35},
    {"name": "Lake Winnebago — Oshkosh", "lat": 44.01, "lng": -88.54, "state": "WI", "type": "lake", "urban": 30},
    # Minnesota
    {"name": "Mississippi River — Minneapolis", "lat": 44.98, "lng": -93.27, "state": "MN", "type": "river", "urban": 70},
    {"name": "Minnesota River — Mankato", "lat": 44.17, "lng": -93.99, "state": "MN", "type": "river", "urban": 30},
    {"name": "St. Croix River — Stillwater", "lat": 45.06, "lng": -92.81, "state": "MN", "type": "river", "urban": 30},
    {"name": "Lake Superior — Duluth", "lat": 46.78, "lng": -92.11, "state": "MN", "type": "lake", "urban": 40},
    {"name": "Red River — Fargo", "lat": 46.88, "lng": -96.79, "state": "MN", "type": "river", "urban": 35},
    # Iowa
    {"name": "Des Moines River — Des Moines", "lat": 41.59, "lng": -93.62, "state": "IA", "type": "river", "urban": 55},
    {"name": "Iowa River — Iowa City", "lat": 41.66, "lng": -91.53, "state": "IA", "type": "river", "urban": 40},
    {"name": "Cedar River — Cedar Rapids", "lat": 41.98, "lng": -91.67, "state": "IA", "type": "river", "urban": 45},
    {"name": "Mississippi River — Davenport", "lat": 41.52, "lng": -90.58, "state": "IA", "type": "river", "urban": 50},
    # Missouri
    {"name": "Missouri River — Kansas City", "lat": 39.10, "lng": -94.58, "state": "MO", "type": "river", "urban": 65},
    {"name": "Mississippi River — St. Louis", "lat": 38.63, "lng": -90.20, "state": "MO", "type": "river", "urban": 70},
    {"name": "Meramec River — Valley Park", "lat": 38.55, "lng": -90.49, "state": "MO", "type": "river", "urban": 45},
    {"name": "Lake of the Ozarks — Osage Beach", "lat": 38.13, "lng": -92.66, "state": "MO", "type": "lake", "urban": 15},
    # North Dakota / South Dakota
    {"name": "Missouri River — Bismarck", "lat": 46.81, "lng": -100.78, "state": "ND", "type": "river", "urban": 25},
    {"name": "Red River — Grand Forks", "lat": 47.93, "lng": -97.03, "state": "ND", "type": "river", "urban": 30},
    {"name": "Missouri River — Pierre", "lat": 44.37, "lng": -100.35, "state": "SD", "type": "river", "urban": 15},
    {"name": "Big Sioux River — Sioux Falls", "lat": 43.55, "lng": -96.73, "state": "SD", "type": "river", "urban": 40},
    # Nebraska / Kansas
    {"name": "Platte River — Omaha", "lat": 41.26, "lng": -95.94, "state": "NE", "type": "river", "urban": 50},
    {"name": "Missouri River — Omaha", "lat": 41.26, "lng": -95.93, "state": "NE", "type": "river", "urban": 55},
    {"name": "Kansas River — Lawrence", "lat": 38.97, "lng": -95.24, "state": "KS", "type": "river", "urban": 35},
    {"name": "Arkansas River — Wichita", "lat": 37.69, "lng": -97.34, "state": "KS", "type": "river", "urban": 45},

    # ---------- SOUTH / GULF ----------
    # Kentucky / Tennessee
    {"name": "Ohio River — Louisville", "lat": 38.25, "lng": -85.76, "state": "KY", "type": "river", "urban": 55},
    {"name": "Kentucky River — Frankfort", "lat": 38.20, "lng": -84.87, "state": "KY", "type": "river", "urban": 30},
    {"name": "Cumberland River — Nashville", "lat": 36.16, "lng": -86.78, "state": "TN", "type": "river", "urban": 55},
    {"name": "Tennessee River — Chattanooga", "lat": 35.05, "lng": -85.31, "state": "TN", "type": "river", "urban": 45},
    {"name": "Tennessee River — Knoxville", "lat": 35.96, "lng": -83.92, "state": "TN", "type": "river", "urban": 45},
    {"name": "Mississippi River — Memphis", "lat": 35.15, "lng": -90.05, "state": "TN", "type": "river", "urban": 55},
    # Alabama / Mississippi
    {"name": "Tennessee River — Decatur", "lat": 34.60, "lng": -86.98, "state": "AL", "type": "river", "urban": 35},
    {"name": "Black Warrior River — Tuscaloosa", "lat": 33.21, "lng": -87.57, "state": "AL", "type": "river", "urban": 35},
    {"name": "Mobile River — Mobile", "lat": 30.69, "lng": -88.04, "state": "AL", "type": "river", "urban": 45},
    {"name": "Coosa River — Gadsden", "lat": 34.01, "lng": -86.01, "state": "AL", "type": "river", "urban": 30},
    {"name": "Pearl River — Jackson", "lat": 32.30, "lng": -90.18, "state": "MS", "type": "river", "urban": 40},
    {"name": "Mississippi River — Vicksburg", "lat": 32.35, "lng": -90.88, "state": "MS", "type": "river", "urban": 30},
    {"name": "Tombigbee River — Columbus", "lat": 33.50, "lng": -88.43, "state": "MS", "type": "river", "urban": 20},
    # Arkansas
    {"name": "Arkansas River — Little Rock", "lat": 34.75, "lng": -92.29, "state": "AR", "type": "river", "urban": 45},
    {"name": "White River — Batesville", "lat": 35.77, "lng": -91.64, "state": "AR", "type": "river", "urban": 15},
    {"name": "Buffalo River — Jasper", "lat": 36.01, "lng": -93.19, "state": "AR", "type": "river", "urban": 5},
    # Louisiana
    {"name": "Mississippi River — New Orleans", "lat": 29.95, "lng": -90.07, "state": "LA", "type": "river", "urban": 70},
    {"name": "Mississippi River — Baton Rouge", "lat": 30.45, "lng": -91.19, "state": "LA", "type": "river", "urban": 55},
    {"name": "Red River — Shreveport", "lat": 32.53, "lng": -93.75, "state": "LA", "type": "river", "urban": 40},
    {"name": "Atchafalaya Basin — Morgan City", "lat": 29.69, "lng": -91.21, "state": "LA", "type": "river", "urban": 15},
    {"name": "Lake Pontchartrain — Mandeville", "lat": 30.36, "lng": -90.07, "state": "LA", "type": "lake", "urban": 45},
    # Texas
    {"name": "Trinity River — Dallas", "lat": 32.78, "lng": -96.80, "state": "TX", "type": "river", "urban": 75},
    {"name": "San Jacinto River — Houston", "lat": 29.76, "lng": -95.37, "state": "TX", "type": "river", "urban": 80},
    {"name": "Colorado River — Austin", "lat": 30.27, "lng": -97.74, "state": "TX", "type": "river", "urban": 55},
    {"name": "Brazos River — Waco", "lat": 31.55, "lng": -97.15, "state": "TX", "type": "river", "urban": 35},
    {"name": "Rio Grande — El Paso", "lat": 31.76, "lng": -106.49, "state": "TX", "type": "river", "urban": 50},
    {"name": "Rio Grande — Laredo", "lat": 27.51, "lng": -99.51, "state": "TX", "type": "river", "urban": 40},
    {"name": "Guadalupe River — San Antonio", "lat": 29.42, "lng": -98.49, "state": "TX", "type": "river", "urban": 55},
    {"name": "Sabine River — Orange", "lat": 30.09, "lng": -93.74, "state": "TX", "type": "river", "urban": 30},
    {"name": "Galveston Bay — Galveston", "lat": 29.30, "lng": -94.80, "state": "TX", "type": "estuary", "urban": 50},
    {"name": "Lake Travis — Austin", "lat": 30.39, "lng": -97.92, "state": "TX", "type": "lake", "urban": 30},
    # Oklahoma
    {"name": "Arkansas River — Tulsa", "lat": 36.15, "lng": -95.99, "state": "OK", "type": "river", "urban": 50},
    {"name": "Canadian River — Oklahoma City", "lat": 35.47, "lng": -97.52, "state": "OK", "type": "river", "urban": 55},
    {"name": "Illinois River — Tahlequah", "lat": 35.92, "lng": -94.97, "state": "OK", "type": "river", "urban": 15},

    # ---------- MOUNTAIN / WEST ----------
    # Colorado
    {"name": "South Platte River — Denver", "lat": 39.74, "lng": -104.99, "state": "CO", "type": "river", "urban": 65},
    {"name": "Fountain Creek — Colorado Springs", "lat": 38.83, "lng": -104.82, "state": "CO", "type": "river", "urban": 50},
    {"name": "Arkansas River — Pueblo", "lat": 38.27, "lng": -104.61, "state": "CO", "type": "river", "urban": 35},
    {"name": "Cache la Poudre River — Fort Collins", "lat": 40.59, "lng": -105.08, "state": "CO", "type": "river", "urban": 40},
    {"name": "Colorado River — Grand Junction", "lat": 39.06, "lng": -108.55, "state": "CO", "type": "river", "urban": 25},
    # Montana / Wyoming / Idaho
    {"name": "Yellowstone River — Billings", "lat": 45.78, "lng": -108.50, "state": "MT", "type": "river", "urban": 30},
    {"name": "Clark Fork — Missoula", "lat": 46.87, "lng": -114.00, "state": "MT", "type": "river", "urban": 30},
    {"name": "Missouri River — Great Falls", "lat": 47.51, "lng": -111.30, "state": "MT", "type": "river", "urban": 25},
    {"name": "North Platte River — Casper", "lat": 42.87, "lng": -106.31, "state": "WY", "type": "river", "urban": 20},
    {"name": "Snake River — Boise", "lat": 43.62, "lng": -116.21, "state": "ID", "type": "river", "urban": 40},
    {"name": "Boise River — Boise", "lat": 43.61, "lng": -116.20, "state": "ID", "type": "river", "urban": 45},
    {"name": "Coeur d'Alene Lake — Coeur d'Alene", "lat": 47.46, "lng": -116.78, "state": "ID", "type": "lake", "urban": 25},
    # Utah / Nevada / New Mexico / Arizona
    {"name": "Jordan River — Salt Lake City", "lat": 40.76, "lng": -111.89, "state": "UT", "type": "river", "urban": 60},
    {"name": "Great Salt Lake — Antelope Island", "lat": 41.06, "lng": -112.24, "state": "UT", "type": "lake", "urban": 20},
    {"name": "Truckee River — Reno", "lat": 39.53, "lng": -119.81, "state": "NV", "type": "river", "urban": 45},
    {"name": "Lake Mead — Boulder City", "lat": 36.02, "lng": -114.77, "state": "NV", "type": "lake", "urban": 25},
    {"name": "Las Vegas Wash — Henderson", "lat": 36.10, "lng": -114.92, "state": "NV", "type": "river", "urban": 70},
    {"name": "Rio Grande — Albuquerque", "lat": 35.08, "lng": -106.65, "state": "NM", "type": "river", "urban": 45},
    {"name": "Salt River — Phoenix", "lat": 33.45, "lng": -111.97, "state": "AZ", "type": "river", "urban": 70},
    {"name": "Gila River — Tucson", "lat": 32.22, "lng": -110.97, "state": "AZ", "type": "river", "urban": 50},
    {"name": "Lake Powell — Page", "lat": 37.07, "lng": -111.49, "state": "AZ", "type": "lake", "urban": 5},

    # ---------- PACIFIC ----------
    # Washington
    {"name": "Duwamish River — Seattle", "lat": 47.53, "lng": -122.34, "state": "WA", "type": "river", "urban": 80},
    {"name": "Puyallup River — Tacoma", "lat": 47.25, "lng": -122.44, "state": "WA", "type": "river", "urban": 55},
    {"name": "Columbia River — Vancouver", "lat": 45.63, "lng": -122.67, "state": "WA", "type": "river", "urban": 50},
    {"name": "Spokane River — Spokane", "lat": 47.66, "lng": -117.43, "state": "WA", "type": "river", "urban": 45},
    {"name": "Yakima River — Yakima", "lat": 46.60, "lng": -120.51, "state": "WA", "type": "river", "urban": 30},
    {"name": "Puget Sound — Everett", "lat": 47.98, "lng": -122.20, "state": "WA", "type": "estuary", "urban": 55},
    {"name": "Lake Washington — Bellevue", "lat": 47.62, "lng": -122.22, "state": "WA", "type": "lake", "urban": 65},
    # Oregon
    {"name": "Willamette River — Portland", "lat": 45.52, "lng": -122.68, "state": "OR", "type": "river", "urban": 65},
    {"name": "Columbia River — The Dalles", "lat": 45.60, "lng": -121.18, "state": "OR", "type": "river", "urban": 25},
    {"name": "Deschutes River — Bend", "lat": 44.06, "lng": -121.31, "state": "OR", "type": "river", "urban": 30},
    {"name": "Rogue River — Grants Pass", "lat": 42.44, "lng": -123.33, "state": "OR", "type": "river", "urban": 25},
    {"name": "Tualatin River — Hillsboro", "lat": 45.52, "lng": -122.99, "state": "OR", "type": "river", "urban": 50},
    # California
    {"name": "San Francisco Bay — Oakland", "lat": 37.80, "lng": -122.27, "state": "CA", "type": "estuary", "urban": 80},
    {"name": "Sacramento River — Sacramento", "lat": 38.58, "lng": -121.49, "state": "CA", "type": "river", "urban": 55},
    {"name": "San Joaquin River — Stockton", "lat": 37.95, "lng": -121.29, "state": "CA", "type": "river", "urban": 45},
    {"name": "Los Angeles River — Los Angeles", "lat": 34.05, "lng": -118.24, "state": "CA", "type": "river", "urban": 90},
    {"name": "Santa Ana River — Riverside", "lat": 33.95, "lng": -117.40, "state": "CA", "type": "river", "urban": 70},
    {"name": "San Diego Creek — Irvine", "lat": 33.67, "lng": -117.83, "state": "CA", "type": "river", "urban": 65},
    {"name": "Lake Tahoe — South Lake Tahoe", "lat": 38.94, "lng": -120.04, "state": "CA", "type": "lake", "urban": 15},
    {"name": "San Gabriel River — Long Beach", "lat": 33.77, "lng": -118.10, "state": "CA", "type": "river", "urban": 80},
    {"name": "Salinas River — Salinas", "lat": 36.67, "lng": -121.66, "state": "CA", "type": "river", "urban": 35},
    {"name": "Russian River — Healdsburg", "lat": 38.61, "lng": -122.87, "state": "CA", "type": "river", "urban": 20},
    {"name": "Klamath River — Yreka", "lat": 41.73, "lng": -122.63, "state": "CA", "type": "river", "urban": 15},
    {"name": "Clear Lake — Lakeport", "lat": 39.04, "lng": -122.92, "state": "CA", "type": "lake", "urban": 15},
    {"name": "Mission Bay — San Diego", "lat": 32.77, "lng": -117.24, "state": "CA", "type": "estuary", "urban": 70},

    # ---------- ALASKA & HAWAII ----------
    {"name": "Ship Creek — Anchorage", "lat": 61.22, "lng": -149.89, "state": "AK", "type": "river", "urban": 50},
    {"name": "Chena River — Fairbanks", "lat": 64.84, "lng": -147.72, "state": "AK", "type": "river", "urban": 30},
    {"name": "Mendenhall River — Juneau", "lat": 58.38, "lng": -134.57, "state": "AK", "type": "river", "urban": 25},
    {"name": "Ala Wai Canal — Honolulu", "lat": 21.28, "lng": -157.83, "state": "HI", "type": "river", "urban": 80},
    {"name": "Pearl Harbor — Ewa Beach", "lat": 21.35, "lng": -157.95, "state": "HI", "type": "estuary", "urban": 55},
    {"name": "Wailuku River — Hilo", "lat": 19.73, "lng": -155.09, "state": "HI", "type": "river", "urban": 25},

    # ==========================================================
    # ADDITIONAL COVERAGE — Major rivers, west coast, sparse areas
    # ==========================================================
    # --- CALIFORNIA (additional) ---
    {"name": "Sacramento River — Red Bluff", "lat": 40.18, "lng": -122.24, "state": "CA", "type": "river", "urban": 25},
    {"name": "Sacramento River — Chico", "lat": 39.72, "lng": -121.85, "state": "CA", "type": "river", "urban": 35},
    {"name": "San Joaquin River — Stockton", "lat": 37.95, "lng": -121.29, "state": "CA", "type": "river", "urban": 50},
    {"name": "San Joaquin River — Fresno", "lat": 36.84, "lng": -119.84, "state": "CA", "type": "river", "urban": 45},
    {"name": "Santa Ana River — Riverside", "lat": 33.95, "lng": -117.40, "state": "CA", "type": "river", "urban": 70},
    {"name": "Russian River — Healdsburg", "lat": 38.62, "lng": -122.87, "state": "CA", "type": "river", "urban": 20},
    {"name": "Klamath River — Yreka", "lat": 41.73, "lng": -122.63, "state": "CA", "type": "river", "urban": 10},
    {"name": "Eel River — Fortuna", "lat": 40.60, "lng": -124.16, "state": "CA", "type": "river", "urban": 10},
    {"name": "Salinas River — Salinas", "lat": 36.67, "lng": -121.66, "state": "CA", "type": "river", "urban": 40},
    {"name": "San Diego Bay — Coronado", "lat": 32.69, "lng": -117.17, "state": "CA", "type": "estuary", "urban": 65},
    {"name": "Mission Bay — San Diego", "lat": 32.78, "lng": -117.23, "state": "CA", "type": "estuary", "urban": 60},
    {"name": "Lake Tahoe — South Shore", "lat": 38.94, "lng": -120.00, "state": "CA", "type": "lake", "urban": 15},
    {"name": "Clear Lake — Lakeport", "lat": 39.04, "lng": -122.91, "state": "CA", "type": "lake", "urban": 15},
    {"name": "Humboldt Bay — Eureka", "lat": 40.77, "lng": -124.19, "state": "CA", "type": "estuary", "urban": 30},
    {"name": "Monterey Bay — Santa Cruz", "lat": 36.96, "lng": -122.01, "state": "CA", "type": "estuary", "urban": 40},

    # --- OREGON (additional) ---
    {"name": "Willamette River — Corvallis", "lat": 44.57, "lng": -123.26, "state": "OR", "type": "river", "urban": 30},
    {"name": "Willamette River — Albany", "lat": 44.63, "lng": -123.10, "state": "OR", "type": "river", "urban": 25},
    {"name": "Columbia River — The Dalles", "lat": 45.61, "lng": -121.18, "state": "OR", "type": "river", "urban": 20},
    {"name": "Deschutes River — Bend", "lat": 44.06, "lng": -121.31, "state": "OR", "type": "river", "urban": 30},
    {"name": "Rogue River — Grants Pass", "lat": 42.44, "lng": -123.33, "state": "OR", "type": "river", "urban": 25},
    {"name": "Umpqua River — Roseburg", "lat": 43.22, "lng": -123.35, "state": "OR", "type": "river", "urban": 20},
    {"name": "Sandy River — Troutdale", "lat": 45.54, "lng": -122.39, "state": "OR", "type": "river", "urban": 35},
    {"name": "Clackamas River — Oregon City", "lat": 45.36, "lng": -122.61, "state": "OR", "type": "river", "urban": 40},
    {"name": "Tualatin River — Tigard", "lat": 45.43, "lng": -122.77, "state": "OR", "type": "river", "urban": 50},
    {"name": "Coos Bay — North Bend", "lat": 43.41, "lng": -124.22, "state": "OR", "type": "estuary", "urban": 20},

    # --- WASHINGTON (additional) ---
    {"name": "Columbia River — Vancouver", "lat": 45.63, "lng": -122.67, "state": "WA", "type": "river", "urban": 50},
    {"name": "Columbia River — Richland", "lat": 46.29, "lng": -119.28, "state": "WA", "type": "river", "urban": 35},
    {"name": "Yakima River — Yakima", "lat": 46.60, "lng": -120.51, "state": "WA", "type": "river", "urban": 35},
    {"name": "Skagit River — Mount Vernon", "lat": 48.42, "lng": -122.33, "state": "WA", "type": "river", "urban": 25},
    {"name": "Snohomish River — Everett", "lat": 47.98, "lng": -122.20, "state": "WA", "type": "river", "urban": 45},
    {"name": "Duwamish River — Tukwila", "lat": 47.47, "lng": -122.26, "state": "WA", "type": "river", "urban": 75},
    {"name": "Puyallup River — Tacoma", "lat": 47.21, "lng": -122.40, "state": "WA", "type": "river", "urban": 55},
    {"name": "Chehalis River — Centralia", "lat": 46.72, "lng": -122.96, "state": "WA", "type": "river", "urban": 20},
    {"name": "Lake Chelan — Chelan", "lat": 47.84, "lng": -120.02, "state": "WA", "type": "lake", "urban": 10},
    {"name": "Bellingham Bay — Bellingham", "lat": 48.75, "lng": -122.50, "state": "WA", "type": "estuary", "urban": 40},

    # --- IDAHO (additional) ---
    {"name": "Snake River — Twin Falls", "lat": 42.56, "lng": -114.46, "state": "ID", "type": "river", "urban": 25},
    {"name": "Snake River — Idaho Falls", "lat": 43.49, "lng": -112.04, "state": "ID", "type": "river", "urban": 30},
    {"name": "Clearwater River — Lewiston", "lat": 46.42, "lng": -117.02, "state": "ID", "type": "river", "urban": 25},
    {"name": "Payette River — Emmett", "lat": 43.87, "lng": -116.50, "state": "ID", "type": "river", "urban": 15},
    {"name": "Lake Coeur d'Alene — Coeur d'Alene", "lat": 47.50, "lng": -116.80, "state": "ID", "type": "lake", "urban": 25},

    # --- MONTANA (additional) ---
    {"name": "Clark Fork River — Missoula", "lat": 46.87, "lng": -114.00, "state": "MT", "type": "river", "urban": 30},
    {"name": "Yellowstone River — Billings", "lat": 45.78, "lng": -108.50, "state": "MT", "type": "river", "urban": 30},
    {"name": "Missouri River — Great Falls", "lat": 47.50, "lng": -111.30, "state": "MT", "type": "river", "urban": 25},
    {"name": "Flathead River — Kalispell", "lat": 48.20, "lng": -114.31, "state": "MT", "type": "river", "urban": 20},

    # --- WYOMING (additional) ---
    {"name": "North Platte River — Casper", "lat": 42.87, "lng": -106.31, "state": "WY", "type": "river", "urban": 25},
    {"name": "Green River — Green River", "lat": 41.53, "lng": -109.47, "state": "WY", "type": "river", "urban": 15},
    {"name": "Snake River — Jackson", "lat": 43.48, "lng": -110.76, "state": "WY", "type": "river", "urban": 15},
    {"name": "Wind River — Riverton", "lat": 43.02, "lng": -108.38, "state": "WY", "type": "river", "urban": 15},

    # --- UTAH (additional) ---
    {"name": "Jordan River — Salt Lake City", "lat": 40.74, "lng": -111.92, "state": "UT", "type": "river", "urban": 65},
    {"name": "Provo River — Provo", "lat": 40.23, "lng": -111.66, "state": "UT", "type": "river", "urban": 40},
    {"name": "Weber River — Ogden", "lat": 41.23, "lng": -111.97, "state": "UT", "type": "river", "urban": 40},
    {"name": "Bear River — Brigham City", "lat": 41.51, "lng": -112.02, "state": "UT", "type": "river", "urban": 20},
    {"name": "Utah Lake — Lehi", "lat": 40.35, "lng": -111.80, "state": "UT", "type": "lake", "urban": 35},

    # --- NEVADA (additional) ---
    {"name": "Truckee River — Reno", "lat": 39.53, "lng": -119.81, "state": "NV", "type": "river", "urban": 55},
    {"name": "Carson River — Carson City", "lat": 39.16, "lng": -119.77, "state": "NV", "type": "river", "urban": 30},
    {"name": "Humboldt River — Winnemucca", "lat": 40.97, "lng": -117.74, "state": "NV", "type": "river", "urban": 10},
    {"name": "Lake Mead — Boulder City", "lat": 36.02, "lng": -114.76, "state": "NV", "type": "lake", "urban": 20},

    # --- ARIZONA (additional) ---
    {"name": "Salt River — Tempe", "lat": 33.43, "lng": -111.94, "state": "AZ", "type": "river", "urban": 70},
    {"name": "Verde River — Scottsdale", "lat": 33.56, "lng": -111.85, "state": "AZ", "type": "river", "urban": 60},
    {"name": "Lake Havasu — Lake Havasu City", "lat": 34.48, "lng": -114.35, "state": "AZ", "type": "lake", "urban": 25},
    {"name": "Lake Powell — Page", "lat": 36.93, "lng": -111.49, "state": "AZ", "type": "lake", "urban": 10},

    # --- NEW MEXICO (additional) ---
    {"name": "Rio Grande — Las Cruces", "lat": 32.35, "lng": -106.75, "state": "NM", "type": "river", "urban": 35},
    {"name": "Rio Grande — Albuquerque", "lat": 35.08, "lng": -106.65, "state": "NM", "type": "river", "urban": 55},
    {"name": "Pecos River — Roswell", "lat": 33.39, "lng": -104.52, "state": "NM", "type": "river", "urban": 20},
    {"name": "San Juan River — Farmington", "lat": 36.73, "lng": -108.21, "state": "NM", "type": "river", "urban": 20},

    # --- COLORADO (additional) ---
    {"name": "South Platte River — Denver", "lat": 39.75, "lng": -104.99, "state": "CO", "type": "river", "urban": 70},
    {"name": "Colorado River — Grand Junction", "lat": 39.07, "lng": -108.55, "state": "CO", "type": "river", "urban": 25},
    {"name": "Arkansas River — Pueblo", "lat": 38.27, "lng": -104.61, "state": "CO", "type": "river", "urban": 35},
    {"name": "Cache la Poudre River — Fort Collins", "lat": 40.59, "lng": -105.08, "state": "CO", "type": "river", "urban": 40},

    # --- NORTH DAKOTA (additional) ---
    {"name": "Red River — Fargo", "lat": 46.88, "lng": -96.79, "state": "ND", "type": "river", "urban": 40},
    {"name": "Missouri River — Bismarck", "lat": 46.81, "lng": -100.78, "state": "ND", "type": "river", "urban": 30},
    {"name": "Lake Sakakawea — Garrison", "lat": 47.65, "lng": -101.42, "state": "ND", "type": "lake", "urban": 5},
    {"name": "James River — Jamestown", "lat": 46.91, "lng": -98.71, "state": "ND", "type": "river", "urban": 15},

    # --- SOUTH DAKOTA (additional) ---
    {"name": "Missouri River — Pierre", "lat": 44.37, "lng": -100.35, "state": "SD", "type": "river", "urban": 15},
    {"name": "Big Sioux River — Sioux Falls", "lat": 43.55, "lng": -96.73, "state": "SD", "type": "river", "urban": 40},
    {"name": "Lake Oahe — Mobridge", "lat": 45.54, "lng": -100.43, "state": "SD", "type": "lake", "urban": 5},
    {"name": "Rapid Creek — Rapid City", "lat": 44.08, "lng": -103.23, "state": "SD", "type": "river", "urban": 30},

    # --- NEBRASKA (additional) ---
    {"name": "Platte River — Grand Island", "lat": 40.92, "lng": -98.34, "state": "NE", "type": "river", "urban": 25},
    {"name": "Missouri River — Omaha", "lat": 41.26, "lng": -95.94, "state": "NE", "type": "river", "urban": 55},
    {"name": "Elkhorn River — Norfolk", "lat": 42.03, "lng": -97.42, "state": "NE", "type": "river", "urban": 20},
    {"name": "Loup River — Columbus", "lat": 41.43, "lng": -97.37, "state": "NE", "type": "river", "urban": 15},

    # --- KANSAS (additional) ---
    {"name": "Kansas River — Lawrence", "lat": 38.97, "lng": -95.24, "state": "KS", "type": "river", "urban": 35},
    {"name": "Arkansas River — Wichita", "lat": 37.69, "lng": -97.34, "state": "KS", "type": "river", "urban": 45},
    {"name": "Smoky Hill River — Salina", "lat": 38.84, "lng": -97.61, "state": "KS", "type": "river", "urban": 20},
    {"name": "Big Blue River — Manhattan", "lat": 39.18, "lng": -96.56, "state": "KS", "type": "river", "urban": 25},

    # --- OKLAHOMA (additional) ---
    {"name": "Arkansas River — Tulsa", "lat": 36.15, "lng": -95.99, "state": "OK", "type": "river", "urban": 50},
    {"name": "Canadian River — Oklahoma City", "lat": 35.42, "lng": -97.52, "state": "OK", "type": "river", "urban": 55},
    {"name": "Red River — Durant", "lat": 33.99, "lng": -96.39, "state": "OK", "type": "river", "urban": 15},
    {"name": "Lake Texoma — Kingston", "lat": 33.96, "lng": -96.72, "state": "OK", "type": "lake", "urban": 10},

    # --- TEXAS (additional) ---
    {"name": "Trinity River — Dallas", "lat": 32.77, "lng": -96.80, "state": "TX", "type": "river", "urban": 75},
    {"name": "Brazos River — Waco", "lat": 31.55, "lng": -97.13, "state": "TX", "type": "river", "urban": 40},
    {"name": "Colorado River — Austin", "lat": 30.27, "lng": -97.74, "state": "TX", "type": "river", "urban": 60},
    {"name": "Guadalupe River — New Braunfels", "lat": 29.70, "lng": -98.13, "state": "TX", "type": "river", "urban": 30},
    {"name": "Nueces River — Corpus Christi", "lat": 27.80, "lng": -97.40, "state": "TX", "type": "river", "urban": 35},
    {"name": "Sabine River — Orange", "lat": 30.09, "lng": -93.74, "state": "TX", "type": "river", "urban": 25},
    {"name": "Rio Grande — El Paso", "lat": 31.76, "lng": -106.44, "state": "TX", "type": "river", "urban": 50},
    {"name": "Rio Grande — Laredo", "lat": 27.51, "lng": -99.51, "state": "TX", "type": "river", "urban": 40},

    # --- MISSISSIPPI (additional) ---
    {"name": "Mississippi River — Vicksburg", "lat": 32.35, "lng": -90.88, "state": "MS", "type": "river", "urban": 30},
    {"name": "Pearl River — Jackson", "lat": 32.30, "lng": -90.18, "state": "MS", "type": "river", "urban": 45},
    {"name": "Tombigbee River — Columbus", "lat": 33.50, "lng": -88.43, "state": "MS", "type": "river", "urban": 20},
    {"name": "Ross Barnett Reservoir — Ridgeland", "lat": 32.43, "lng": -90.05, "state": "MS", "type": "lake", "urban": 30},

    # --- ARKANSAS (additional) ---
    {"name": "Arkansas River — Fort Smith", "lat": 35.39, "lng": -94.40, "state": "AR", "type": "river", "urban": 35},
    {"name": "White River — Batesville", "lat": 35.77, "lng": -91.64, "state": "AR", "type": "river", "urban": 15},
    {"name": "Ouachita River — Hot Springs", "lat": 34.50, "lng": -93.06, "state": "AR", "type": "river", "urban": 25},
    {"name": "Buffalo River — Yellville", "lat": 36.23, "lng": -92.68, "state": "AR", "type": "river", "urban": 5},

    # --- LOUISIANA (additional) ---
    {"name": "Mississippi River — Baton Rouge", "lat": 30.45, "lng": -91.19, "state": "LA", "type": "river", "urban": 55},
    {"name": "Red River — Shreveport", "lat": 32.53, "lng": -93.75, "state": "LA", "type": "river", "urban": 40},
    {"name": "Atchafalaya River — Morgan City", "lat": 29.69, "lng": -91.21, "state": "LA", "type": "river", "urban": 15},
    {"name": "Bayou Lafourche — Thibodaux", "lat": 29.80, "lng": -90.82, "state": "LA", "type": "river", "urban": 20},

    # --- KENTUCKY (additional) ---
    {"name": "Kentucky River — Frankfort", "lat": 38.20, "lng": -84.87, "state": "KY", "type": "river", "urban": 30},
    {"name": "Licking River — Covington", "lat": 39.08, "lng": -84.51, "state": "KY", "type": "river", "urban": 45},
    {"name": "Green River — Bowling Green", "lat": 36.99, "lng": -86.44, "state": "KY", "type": "river", "urban": 30},
    {"name": "Cumberland River — Williamsburg", "lat": 36.74, "lng": -84.16, "state": "KY", "type": "river", "urban": 15},

    # --- WEST VIRGINIA (additional) ---
    {"name": "Kanawha River — Charleston", "lat": 38.35, "lng": -81.63, "state": "WV", "type": "river", "urban": 45},
    {"name": "Monongahela River — Morgantown", "lat": 39.63, "lng": -79.96, "state": "WV", "type": "river", "urban": 35},
    {"name": "Greenbrier River — Lewisburg", "lat": 37.80, "lng": -80.44, "state": "WV", "type": "river", "urban": 10},

    # --- DELAWARE (additional) ---
    {"name": "Christina River — Wilmington", "lat": 39.74, "lng": -75.55, "state": "DE", "type": "river", "urban": 60},
    {"name": "Broadkill River — Milton", "lat": 38.78, "lng": -75.31, "state": "DE", "type": "river", "urban": 15},
    {"name": "Indian River Bay — Rehoboth Beach", "lat": 38.61, "lng": -75.07, "state": "DE", "type": "estuary", "urban": 25},

    # ==========================================================
    # MAJOR RIVERS — extra points along the Mississippi, Ohio, Missouri, etc.
    # Provides backbone coverage across the country
    # ==========================================================
    {"name": "Mississippi River — Memphis", "lat": 35.15, "lng": -90.05, "state": "TN", "type": "river", "urban": 55},
    {"name": "Mississippi River — St. Louis", "lat": 38.63, "lng": -90.20, "state": "MO", "type": "river", "urban": 60},
    {"name": "Mississippi River — Quad Cities", "lat": 41.52, "lng": -90.58, "state": "IA", "type": "river", "urban": 40},
    {"name": "Mississippi River — La Crosse", "lat": 43.80, "lng": -91.25, "state": "WI", "type": "river", "urban": 30},
    {"name": "Mississippi River — New Orleans", "lat": 29.95, "lng": -90.07, "state": "LA", "type": "river", "urban": 70},
    {"name": "Mississippi River — Natchez", "lat": 31.56, "lng": -91.40, "state": "MS", "type": "river", "urban": 20},
    {"name": "Ohio River — Cincinnati", "lat": 39.10, "lng": -84.51, "state": "OH", "type": "river", "urban": 60},
    {"name": "Ohio River — Louisville", "lat": 38.26, "lng": -85.76, "state": "KY", "type": "river", "urban": 55},
    {"name": "Ohio River — Evansville", "lat": 37.97, "lng": -87.57, "state": "IN", "type": "river", "urban": 40},
    {"name": "Ohio River — Wheeling", "lat": 40.07, "lng": -80.72, "state": "WV", "type": "river", "urban": 35},
    {"name": "Missouri River — Kansas City", "lat": 39.10, "lng": -94.58, "state": "MO", "type": "river", "urban": 55},
    {"name": "Missouri River — Sioux City", "lat": 42.50, "lng": -96.40, "state": "IA", "type": "river", "urban": 30},
    {"name": "Missouri River — Yankton", "lat": 42.87, "lng": -97.39, "state": "SD", "type": "river", "urban": 15},
    {"name": "Tennessee River — Chattanooga", "lat": 35.05, "lng": -85.31, "state": "TN", "type": "river", "urban": 45},
    {"name": "Tennessee River — Knoxville", "lat": 35.96, "lng": -83.92, "state": "TN", "type": "river", "urban": 50},
    {"name": "Cumberland River — Nashville", "lat": 36.17, "lng": -86.78, "state": "TN", "type": "river", "urban": 55},
    {"name": "Savannah River — Augusta", "lat": 33.47, "lng": -81.97, "state": "GA", "type": "river", "urban": 35},
    {"name": "Savannah River — Savannah", "lat": 32.08, "lng": -81.09, "state": "GA", "type": "river", "urban": 40},
    {"name": "James River — Richmond", "lat": 37.53, "lng": -77.43, "state": "VA", "type": "river", "urban": 55},
    {"name": "Potomac River — Washington DC", "lat": 38.90, "lng": -77.04, "state": "VA", "type": "river", "urban": 80},
    {"name": "Hudson River — Poughkeepsie", "lat": 41.70, "lng": -73.93, "state": "NY", "type": "river", "urban": 35},
    {"name": "Hudson River — Troy", "lat": 42.73, "lng": -73.69, "state": "NY", "type": "river", "urban": 40},
    {"name": "Delaware River — Trenton", "lat": 40.22, "lng": -74.76, "state": "NJ", "type": "river", "urban": 50},
    {"name": "Susquehanna River — Harrisburg", "lat": 40.26, "lng": -76.88, "state": "PA", "type": "river", "urban": 45},
    {"name": "Allegheny River — Pittsburgh", "lat": 40.45, "lng": -79.98, "state": "PA", "type": "river", "urban": 60},
    {"name": "Wabash River — Terre Haute", "lat": 39.47, "lng": -87.41, "state": "IN", "type": "river", "urban": 30},
    {"name": "Illinois River — Peoria", "lat": 40.69, "lng": -89.59, "state": "IL", "type": "river", "urban": 35},
    {"name": "Des Moines River — Des Moines", "lat": 41.59, "lng": -93.62, "state": "IA", "type": "river", "urban": 45},
    {"name": "Cedar River — Cedar Rapids", "lat": 42.00, "lng": -91.64, "state": "IA", "type": "river", "urban": 40},
    {"name": "Minnesota River — Mankato", "lat": 44.17, "lng": -94.00, "state": "MN", "type": "river", "urban": 25},
    {"name": "St. Croix River — Stillwater", "lat": 45.06, "lng": -92.81, "state": "MN", "type": "river", "urban": 25},
    {"name": "Fox River — Green Bay", "lat": 44.52, "lng": -88.02, "state": "WI", "type": "river", "urban": 40},
    {"name": "Rock River — Rockford", "lat": 42.27, "lng": -89.09, "state": "IL", "type": "river", "urban": 35},
]


# Known PFAS hotspots for proximity-based contamination modeling
_HOTSPOTS = [
    {"lat": 35.05, "lng": -78.88, "intensity": 1.0, "name": "Cape Fear NC"},
    {"lat": 34.60, "lng": -86.98, "intensity": 0.9, "name": "Decatur AL (3M)"},
    {"lat": 39.27, "lng": -81.56, "intensity": 0.85, "name": "Parkersburg WV (DuPont)"},
    {"lat": 44.45, "lng": -83.33, "intensity": 0.8, "name": "Oscoda MI (AFFF)"},
    {"lat": 42.88, "lng": -73.20, "intensity": 0.75, "name": "Bennington VT"},
    {"lat": 42.33, "lng": -85.57, "intensity": 0.7, "name": "Parchment MI"},
    {"lat": 42.90, "lng": -73.35, "intensity": 0.7, "name": "Hoosick Falls NY"},
    {"lat": 41.50, "lng": -74.01, "intensity": 0.65, "name": "Newburgh NY (AFFF)"},
    {"lat": 40.18, "lng": -75.13, "intensity": 0.7, "name": "Horsham PA (AFFF)"},
    {"lat": 38.80, "lng": -104.72, "intensity": 0.6, "name": "Colorado Springs (AFFF)"},
    {"lat": 42.86, "lng": -71.49, "intensity": 0.65, "name": "Merrimack NH"},
    {"lat": 47.53, "lng": -122.34, "intensity": 0.55, "name": "Seattle WA"},
    {"lat": 37.80, "lng": -122.40, "intensity": 0.5, "name": "San Francisco CA"},
    {"lat": 41.89, "lng": -87.63, "intensity": 0.5, "name": "Chicago IL"},
    {"lat": 40.74, "lng": -74.17, "intensity": 0.6, "name": "Newark NJ (Passaic)"},
    {"lat": 29.76, "lng": -95.37, "intensity": 0.45, "name": "Houston TX"},
    {"lat": 34.05, "lng": -118.24, "intensity": 0.4, "name": "Los Angeles CA"},
    {"lat": 33.45, "lng": -111.97, "intensity": 0.35, "name": "Phoenix AZ"},
    {"lat": 44.98, "lng": -93.27, "intensity": 0.45, "name": "Minneapolis MN (3M HQ)"},
    {"lat": 30.45, "lng": -87.19, "intensity": 0.5, "name": "Pensacola FL (NAS)"},
    {"lat": 28.01, "lng": -82.46, "intensity": 0.4, "name": "Tampa FL (MacDill AFB)"},
]


def _haversine(lat1, lng1, lat2, lng2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lng1, lat2, lng2])
    dlat = la2 - la1
    dlng = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def generate_synthetic_water_pfas(wb):
    """
    Generate a plausible water PFAS concentration (ng/L) for a water body
    based on proximity to known hotspots and urban land cover.
    Returns a value that looks realistic on the map.
    """
    rng = np.random.RandomState(hash(wb['name']) % (2**31))

    # Base: log-normal background centered at 2 ng/L (national median from UCMR5)
    base = rng.lognormal(mean=np.log(2.0), sigma=0.6)

    # Urban contribution: more urban = more PFAS from WWTP effluent
    urban_boost = wb['urban'] / 100.0 * rng.uniform(1.5, 6.0)

    # Hotspot proximity: exponential decay with distance
    hotspot_boost = 0.0
    for hs in _HOTSPOTS:
        dist = _haversine(wb['lat'], wb['lng'], hs['lat'], hs['lng'])
        if dist < 200:  # within 200 km
            contribution = hs['intensity'] * 80 * math.exp(-dist / 40)
            hotspot_boost += contribution

    # Combine
    water_pfas = base + urban_boost + hotspot_boost

    # Add noise
    water_pfas *= rng.uniform(0.7, 1.4)

    # Clip to realistic range
    return round(float(np.clip(water_pfas, 0.3, 800)), 2)
