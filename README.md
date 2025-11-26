[![PyPI version fury.io](https://badge.fury.io/py/focalpy.svg)](https://pypi.python.org/pypi/focalpy/)
[![Documentation Status](https://readthedocs.org/projects/focalpy/badge/?version=latest)](https://focalpy.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/focalpy/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/focalpy/main)
[![codecov](https://codecov.io/gh/martibosch/focalpy/branch/main/graph/badge.svg?token=hKoSSRn58a)](https://codecov.io/gh/martibosch/focalpy)
[![GitHub license](https://img.shields.io/github/license/martibosch/focalpy.svg)](https://github.com/martibosch/focalpy/blob/main/LICENSE)

# focalpy

Toolkit for focal site multi-scale studies in Python.

![stations-tree-canopy](https://github.com/martibosch/focalpy/raw/main/figures/stations-tree-canopy.png)

*(C) OpenStreetMap contributors, tiles style by Humanitarian OpenStreetMap Team hosted by OpenStreetMap France*

## Overview

Compute multi-scale spatial predictors:

```python
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn import ensemble

import focalpy

species_richness_filepath = "data/bird-richness.gpkg"
buildings_gdf_filepath = "data/buildings.gpkg"
tree_canopy_filepath = "data/tree-canopy.tif"

buffer_dists = [100, 250, 500]

species_gdf = gpd.read_file(species_richness_filepath)
y_col = "n.species"  # species richness

fa = focalpy.FocalAnalysis(
    [buildings_gdf_filepath, tree_canopy_filepath],
    species_gdf,
    buffer_dists,
    [
        "compute_vector_features",
        "compute_raster_features",
    ],
    feature_col_prefixes=["building", "tree"],
    feature_methods_args={
        "compute_vector_features": [{"area": "sum"}],
    },
    feature_methods_kwargs={
        "compute_raster_features": {"stats": "sum"},
    },
)
fa.features_df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>building_area_sum_100</th>
      <th>building_area_sum_250</th>
      <th>building_area_sum_500</th>
      <th>tree_sum_100</th>
      <th>tree_sum_250</th>
      <th>tree_sum_500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13069.227511</td>
      <td>60218.251616</td>
      <td>207368.012055</td>
      <td>2016.0</td>
      <td>14875.0</td>
      <td>61452.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7439.635337</td>
      <td>41645.546860</td>
      <td>131432.855040</td>
      <td>1331.0</td>
      <td>15760.0</td>
      <td>84520.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8962.495280</td>
      <td>54251.129360</td>
      <td>146157.281494</td>
      <td>2385.0</td>
      <td>16725.0</td>
      <td>79704.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8001.653873</td>
      <td>29735.393494</td>
      <td>102803.559740</td>
      <td>2512.0</td>
      <td>22892.0</td>
      <td>95945.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10447.531020</td>
      <td>39405.263870</td>
      <td>110922.947475</td>
      <td>3886.0</td>
      <td>19860.0</td>
      <td>99111.0</td>
    </tr>
  </tbody>
</table>

```
# target area (for region-wide prediction/extrapolation)
study_area_filepath = "data/study-area.gpkg"
grid_res = 500

# train a model and spatially extrapolate it
model = ensemble.GradientBoostingRegressor().fit(fa.features_df, species_gdf[y_col])
pred_da = fa.predict_raster(model, study_area_filepath, grid_res, pred_label=y_col)

# plot the field data and predicted raster
fig, ax = plt.subplots()
cmap = "BuGn"
vmin = min(pred_da.min().item(), species_gdf[y_col].min())
vmax = max(pred_da.max().item(), species_gdf[y_col].max())
pred_da.plot(ax=ax, alpha=0.7, vmin=vmin, vmax=vmax, cmap=cmap)
species_gdf.plot(y_col, ax=ax, edgecolor="k", vmin=vmin, vmax=vmax, cmap=cmap)
ax.set_axis_off()
cx.add_basemap(ax, crs=species_gdf.crs, attribution=False)
```

![pred-raster](https://github.com/martibosch/focalpy/raw/main/figures/pred-raster.png)

*(C) OpenStreetMap contributors, tiles style by Humanitarian OpenStreetMap Team hosted by OpenStreetMap France*

See the [user guide](https://focalpy.readthedocs.io/en/latest/user-guide/introduction.html) and the [API documentation](https://focalpy.readthedocs.io/en/latest/api.html) for more details on the features of focalpy.

## Installation

Like many other geospatial Python packages, focalpy requires many base C libraries that cannot be installed with pip. Accordingly, the best way to install focalpy is to use conda/mamba, i.e., in a given conda environment, run:

```bash
# or mamba install -c conda-forge geopandas
conda install -c conda-forge geopandas
```

Within the same conda environment, you can then install focalpy using pip:

```bash
pip install https://github.com/martibosch/focalpy/archive/main.zip
```

## Acknowledgements

- This package was created with the [martibosch/cookiecutter-geopy-package](https://github.com/martibosch/cookiecutter-geopy-package) project template.

## References

1. Huais, P. Y. (2024). Multilandr: An r package for multi-scale landscape analysis. Landscape Ecology, 39(8), 140.
