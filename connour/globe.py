# This is how to plot a polygon on a globe!!

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

ax = plt.axes(projection=ccrs.NearsidePerspective(central_latitude=40, central_longitude=-105, satellite_height=35785831))

ax.set_global()
patches = []
# poly should be lon, lat
poly = np.array([[-105, 60], [-110, 40], [-100, 40], [-90, 30]])
# I have absolutely no clue why it takes PlateCarree as the transform instead of NearsidePerspective but it works!!!
polygon = Polygon(poly, closed=True, color=(0.0, 1.0, 0.5), transform=ccrs.PlateCarree())
ax.add_patch(polygon)
ax.coastlines()
ax.gridlines()

plt.show()
