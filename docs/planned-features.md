# Planned features

- Support spatial regression models from [spreg](https://github.com/pysal/spreg) and the PySAL stack {cite:p}`rey2009pysal`.
- Add a method to plot *scalograms*, i.e., plotting how the computed spatial predictors respond to changes in scale, which can reveal scale thresholds that maximize landscape heterogeneity {cite:p}`pasher2013optimizing` (and therefore the variance of the spatial predictors that act as independent variables).
- Implement algorithms to sample locations for field data collection based on landscape heterogeneity {cite:p}`bowler2022optimising`.
- Add methods to assess the "area of applicability" {cite:p}`meyer2021predicting` of the models (based on the latent space defined by the spatial predictors) as well as the "risk of spatial extrapolation" {cite:p}`gutzwiller2023using`.
