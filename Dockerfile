FROM jupyter/datascience-notebook
MAINTAINER Richard Fergie <richard.fergie@gmail.com>

RUN pip install pystan jupyterstan convertdate lunarcalendar holidays plotly
# Fix for https://github.com/facebook/prophet/issues/1598
# Need to have dependencies installed first
RUN pip install fbprophet
