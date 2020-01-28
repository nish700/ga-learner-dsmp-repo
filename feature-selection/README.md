### Project Overview

 Feature selction is process of selecting subset of relevant features for use in model construction. The data contains some features which are either redundant or irrelevant and can be removed without incurring much loss of information. Some features are strongly related to output while others are strongly related to other features.
The one's strongly related to the labels(outputs) are retained while the one's strongly related to other features are removed to prevent the model from learning such direct relationship.

In the project , the model is tested against different set of features and the best set of features is finally used for the prediction.

**About the Dataset:**

The data set (15120 observations) contains both features and the Cover_Type.

**Feature	Description**

**Elevation** Elevation in meters

**Aspect**  Aspect in degrees azimuth

**Slope**	Slope in degrees

**Horizontal Distance To Hydrology**	Horz Dist to nearest surface water features

**Vertical Distance To Hydrology**	Vert Dist to nearest surface water features

**Horizontal Distance To Roadways**	Horz Dist to nearest roadway

**Hillshade_9am (0 to 255 index)**	Hillshade index at 9am, summer solstice

**Hillshade_Noon (0 to 255 index)**	Hillshade index at noon, summer solstice

**Hillshade_3pm (0 to 255 index)**	Hillshade index at 3pm, summer solstice

**Horizontal Distance To Fire Points**	Horizontal Dist to nearest wildfire ignition points

**Wilderness_Area (4 binary columns, 0 = absence or 1 = presence)**	Wilderness area designation

**Soil_Type (40 binary columns, 0 = absence or 1 = presence)**	Soil Type designation

**Cover_Type (7 types, integers 1 to 7)**	Forest Cover Type designation





**Following concepts have been implemented in the project:**

•	How are the features important to our model.

•	How to select the most significant features out of many.

•	How to perform univariate feature selection.

•	How to perform a multivariate feature selection.



