# ProphetDAG

[Prophet](https://facebook.github.io/prophet/) allows you to add regressors to
your forecast to help better predict the future. When adding a regressor you
must know the values in the past (for training) and in the future (for
prediction).

But often you want to use something as a regressor where you *don't* know what
the future values will be. The Prophet documentation [has this to
say](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#additional-regressors)
about this:

> One can also use as a regressor another time series that has been forecasted
> with a time series model, such as Prophet. For instance, if r(t) is included
> as a regressor for y(t), Prophet can be used to forecast r(t) and then that
> forecast can be plugged in as the future values when forecasting y(t). A note
> of caution around this approach: This will probably not be useful unless r(t)
> is somehow easier to forecast then y(t). This is because error in the forecast
> of r(t) will produce error in the forecast of y(t)

Prophet does not propagate the uncertainty for `r(t)` into the forecast for
`y(t)`; this is where `ProphetDAG` can help.

See the [notebook](blob/master/ProphetDAG demo.ipynb) for a worked example.

The key parts of the example are as follows:

## Define the DAG

```
from prophetDAG import ProphetDAG
import networkx as nx
from fbprophet import Prophet

graph = nx.DiGraph()
graph.add_nodes_from([(1,{'name':'total'})
                     ,(2,{'name':'Hamilton'})
                     ,(3,{'name':'Washington'})
                     ,(4,{'name':'Franklin'})])
graph.add_edge(2,1)
graph.add_edge(3,1)
graph.add_edge(4,1)
```

## Create a Prophet Model for each node and attach the data

```
for (i,d) in zip(graph.nodes,[total,hamilton,washington,franklin]):
    graph.nodes[i]['df'] = d
    m = Prophet()
    m.add_country_holidays(country_name="US")
    m.history_dates = history_dates
    m.start = start
    m.t_scale = t_scale
    future = m.make_future_dataframe(periods=365)
    graph.nodes[i]['prophet'] = m
    graph.nodes[i]['future'] = future
```

## Fit the model

```
p = ProphetDAG()
result = p.fit(graph)
```
