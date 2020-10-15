import networkx as nx
from fbprophet import Prophet
import pandas as pd
import numpy as np
import pystan

class ProphetDAG(object):
    def __init__(self, n_samp=100):
        self.n_samp = n_samp

    functions_block = """
functions {
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    matrix[T, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx += 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  vector logistic_gamma(real k, real m, vector delta, vector t_change, int S) {
    vector[S] gamma;  // adjusted offsets, for piecewise continuity
    vector[S + 1] k_s;  // actual rate in each segment
    real m_pr;

    // Compute the rate in each segment
    k_s[1] = k;
    for (i in 1:S) {
      k_s[i + 1] = k_s[i] + delta[i];
    }

    // Piecewise offsets
    m_pr = m; // The offset in the previous segment
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // update for the next segment
    }
    return gamma;
  }

  vector logistic_trend(
    real k, real m, vector delta, vector t, vector cap, matrix A,
    vector t_change, int S
  ) {
    vector[S] gamma;

    gamma = logistic_gamma(k, m, delta, t_change, S);
    return cap ./ (1 + exp(-(k + A * delta) .* (t - (m + A * gamma))));
  }

  // Linear trend function

  vector linear_trend(
    real k, real m, vector delta, vector t, matrix A, vector t_change
  ) {
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }

  // Flat trend function

  vector flat_trend(
    real m,
    int T
  ) {
    return rep_vector(m, T);
  }

  // Helper for getting appropriate trend

  vector get_trend(
    real k, real m, vector delta, vector t, vector cap, matrix A,
    vector t_change, int S, int trend_indicator, int T
  ) {
    if (trend_indicator == 0) {
      return linear_trend(k, m, delta, t, A, t_change);
    } else if (trend_indicator==1) {
      return logistic_trend(k, m, delta, t, cap, A, t_change, S);
    } else {
      return flat_trend(m, T);
    }
  }
}
"""

    data_block_constant = """
int T;                // Number of time periods
vector[T] t;          // Time
int T_pred;           // Number of prediction time periods
vector[T_pred] t_pred; // times for predictions
int n_samp; // Number of samples for trend uncertainty
"""

    def data_block(self, i, n_nodes):
      str = f"""
int<lower=1> K_{i};       // Number of regressors
vector[T] cap_{i};        // Capacities for logistic trend
vector[T] y_{i};          // Time series
int S_{i};                // Number of changepoints
vector[S_{i}] t_change_{i};   // Times of trend changepoints
matrix[T,K_{i}] X_{i};        // Regressors
vector[K_{i}] sigmas_{i};     // Scale on seasonality prior
real<lower=0> tau_{i};    // Scale on changepoints prior
int trend_indicator_{i};  // 0 for linear, 1 for logistic, 2 for flat
vector[K_{i}] s_a_{i};        // Indicator of additive features
vector[K_{i}] s_m_{i};        // Indicator of multiplicative features
vector[{n_nodes}] a_{i};
vector[{n_nodes}] m_{i};
vector[T_pred] cap_pred_{i};
matrix[T_pred, K_{i}] X_pred_{i};
int S_pred_{i}; // Upper bound on number of future changepoints
    """
      return(str)

    def transformed_data_decleration(self, i):
      str = f"""
    matrix[T, S_{i}] A_{i};
      """
      return(str)

    def transformed_data_define(self, i):
      str=f"""
    A_{i} = get_changepoint_matrix(t, t_change_{i}, T, S_{i});
      """
      return(str)

    def parameters_block(self, i,parents):
      n_parents = len(parents)
      str = f"""
real k_{i};                   // Base trend growth rate
real offset_{i};              // Trend offset
vector[S_{i}] delta_{i};      // Trend rate adjustments
real<lower=0> sigma_obs_{i};  // Observation noise
vector[K_{i}+{n_parents}] beta_{i};       // Regressor coefficients
      """
      return(str)

    def transformed_params_decleration(self, i):
      str = f"""
    vector[T] trend_{i};
      """
      return(str)

    def transformed_params_define(self, i):
      # Double braces {{ }} are for escaping in f-strings
      str = f"""
trend_{i} = get_trend(k_{i}, offset_{i}, delta_{i}, t, cap_{i}, A_{i}, t_change_{i}, S_{i}, trend_indicator_{i}, T);
      """
      return(str)

    def append_row_sigmas(self, i, parents):
      if len(parents)==0:
        str=f"sigmas_{i}"
      else:
        if parents[0]:
            sig = parents[0]
        else:
            sig = 10
        rest = self.append_row_sigmas(i,parents[1:])
        str=f"append_row({rest}, {sig})"
      return(str)


    def model_priors(self, i, parents):
      str = f"""
k_{i} ~ normal(0, 5);
offset_{i} ~ normal(0, 5);
delta_{i} ~ double_exponential(0, tau_{i});
sigma_obs_{i} ~ normal(0, 0.5);
beta_{i} ~ normal(0, {self.append_row_sigmas(i,parents)});
    """
      return(str)

    # recursive function to help build up nested append_col
    # parents is a list of numeric nodeids
    def append_col(self, x, y, y_post, i, parents):
      if len(parents)==0:
        str=f"{x}{i}"
      else:
        rest = self.append_col(x,y,y_post,i,parents[1:])
        str=f"append_col({rest},{y}{parents[0]}{y_post})"
      return(str)

    # recursive function to build up nested append_row for
    # feature_type is a/m
    def append_row(self, feature_type,i, parents):
      if len(parents)==0:
        str=f"s_{feature_type}_{i}"
      else:
        rest = self.append_row(feature_type,i,parents[1:])
        str=f"append_row({rest},{feature_type}_{i}[{parents[0]}])"
      return(str)

    def model_likelihood(self, i, parents):
      X = self.append_col("X_","y_","",i,parents)
      s_m = self.append_row("m",i,parents)
      s_a = self.append_row("a",i,parents)
      str=f"""
y_{i} ~ normal(
trend_{i}
.* (1 + {X} * (beta_{i} .* {s_m}))
+ {X} * (beta_{i} .* {s_a}),
sigma_obs_{i}
);
      """
      return(str)

    def generated_quantities_declare(self,i):
        str = f"""
vector[T_pred] y_hat_{i};
vector[T_pred] trend_hat_{i};
matrix[T_pred, S_{i}] A_pred_{i};
matrix[T_pred, n_samp] trend_samples_{i};
matrix[T_pred, n_samp] y_pred_{i};
vector[S_1 + S_pred_{i}] t_change_sim_{i};
vector[S_1 + S_pred_{i}] delta_sim_{i};
real lambda_{i};
matrix[T_pred, S_{i} + S_pred_{i}] A_sim_{i};
        """
        return(str)

    def generated_quantities_estimate(self, i, parents):
        X = self.append_col("X_pred_","y_hat_","",i,parents)
        s_m = self.append_row("m",i,parents)
        s_a = self.append_row("a",i,parents)
        str = f"""
A_pred_{i} = get_changepoint_matrix(t_pred, t_change_{i}, T_pred, S_{i});
trend_hat_{i} = get_trend(
      k_{i}, offset_{i}, delta_{i}, t_pred, cap_pred_{i}, A_pred_{i}, t_change_{i}, S_{i}, trend_indicator_{i}, T_pred
      );

y_hat_{i} = trend_hat_{i} .* (1 + {X} * (beta_{i} .* {s_m}))
      + {X} * (beta_{i} .* {s_a});

for (i in 1:S_{i}) {{
      t_change_sim_{i}[i] = t_change_{i}[i];
      delta_sim_{i}[i] = delta_{i}[i];
}}

lambda_{i} = mean(fabs(delta_{i})) + 1e-8;
        """
        return(str)

    def generate_quantities_sampling(self,i,parents):
        X = self.append_col("X_pred_","y_pred_","[:,i]",i,parents)
        s_m = self.append_row("m",i,parents)
        s_a = self.append_row("a",i,parents)
        str = f"""
if (S_pred_1 > 0) {{
   //Sample new changepoints from a Poisson process with rate S
   //Sample changepoint deltas from Laplace(lambda)
   t_change_sim_{i}[S_{i} + 1] = 1 + exponential_rng(S_{i});
   for (j in (S_{i} + 2):(S_{i} + S_pred_{i})) {{
          t_change_sim_{i}[j] = t_change_sim_{i}[j - 1] + exponential_rng(S_{i});
   }}
   for (j in (S_{i} + 1): (S_{i} + S_pred_{i})) {{
          delta_sim_{i}[j] = double_exponential_rng(0, lambda_{i});
   }}
}}
// Compute trend with these changepoints
A_sim_{i} = get_changepoint_matrix(t_pred, t_change_sim_{i}, T_pred, S_{i} + S_pred_{i});
trend_samples_{i}[:, i] = get_trend(
        k_{i}, offset_{i}, delta_sim_{i}, t_pred, cap_pred_{i}, A_sim_{i}, t_change_sim_{i}, S_{i} + S_pred_{i},
        trend_indicator_{i}, T_pred
      );
y_pred_{i}[:,i] = trend_samples_{i}[:, i] .* (1 + {X} * (beta_{i} .* {s_m}))
        + {X} * (beta_{i} .* {s_a});
        """
        return(str)

    def generate_stan_code(self, graph):
        nodes = list(nx.topological_sort(graph))
        n_nodes = len(nodes)
        data = [self.data_block_constant]
        transformed_data_declare = []
        transformed_data_defines = []
        parameters = []
        transformed_parameters_declare = []
        transformed_parameters_defines = []
        model = []
        generated_quantities_declares = []
        generated_quantities_estimate = []
        generated_quantities_sampling = []
        for i in nodes:
          parents = list(graph.predecessors(i))
          prior_scales = []
          for j in parents:
            edge = graph[j][i]
            prior_scales.append(edge.get('prior_scale'))
          data.append(self.data_block(i,n_nodes))
          transformed_data_declare.append(self.transformed_data_decleration(i))
          transformed_data_defines.append(self.transformed_data_define(i))
          parameters.append(self.parameters_block(i,parents))
          transformed_parameters_declare.append(self.transformed_params_decleration(i))
          transformed_parameters_defines.append(self.transformed_params_define(i))
          model.append(self.model_priors(i,prior_scales))
          model.append(self.model_likelihood(i,parents))
          generated_quantities_declares.append(self.generated_quantities_declare(i))
          generated_quantities_estimate.append(self.generated_quantities_estimate(i,parents))
          generated_quantities_sampling.append(self.generate_quantities_sampling(i,parents))
        nl = "\n" # f-strings don't allow \
        str = f"""
{self.functions_block}
data {{
  {nl.join(data)}
}}
transformed data {{
  {nl.join(transformed_data_declare)}
  {nl.join(transformed_data_defines)}
}}
parameters {{
  {nl.join(parameters)}
}}
transformed parameters {{
  {nl.join(transformed_parameters_declare)}
  {nl.join(transformed_parameters_defines)}
}}
model {{
  {nl.join(model)}
}}
generated quantities {{
  {nl.join(generated_quantities_declares)}

  if(T_pred > 0) {{
  {nl.join(generated_quantities_estimate)}
  for (i in 1:n_samp) {{
        {nl.join(generated_quantities_sampling)}
  }}
  }}
}}
        """
        return(str)

    def fit(self, graph):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Input graph must be a directed acyclic graph (DAG)")
        all_dat = {}
        all_init = {}
        for i in graph.nodes:
            node = graph.nodes[i]
            parents = list(graph.predecessors(i))
            if node.get("prophet") is None:
                raise ValueError("All nodes must have an attribute called 'prophet' contained a prophet object")
            m = graph.nodes[i]["prophet"]
            if node.get("df") is None:
                raise ValueError("All nodes must have an attribute called 'df' containing a pandas DataFrame")
            if node.get("future") is None:
                raise ValueError("All nodes must have an attribute called 'future' containing a pandas DataFrame")
            df = graph.nodes[i]["df"]
            future = graph.nodes[i]["future"]
            future = m.setup_dataframe(future.copy())
            only_future = future[df.shape[0]:]
            history = df[df['y'].notnull()].copy()
            m.history_dates = pd.to_datetime(pd.Series(df['ds'].unique(), name='ds')).sort_values()
            history = m.setup_dataframe(history, initialize_scales=True)
            m.history = history
            m.set_auto_seasonalities()
            seasonal_features, prior_scales, component_cols, modes = (
                m.make_all_seasonality_features(history))
            m.train_component_cols = component_cols
            m.component_modes = modes
            m.set_changepoints()
            trend_indicator = {'linear': 0, 'logistic': 1, 'flat': 2}
            seasonal_features_future, _, _, _ = (
               m.make_all_seasonality_features(future)
            )
            dat = {
                'T': m.history.shape[0],
                'T_pred': only_future.shape[0],
                't_pred': np.array(only_future.t),
                f'K_{i}': seasonal_features.shape[1],
                f'S_{i}': len(m.changepoints_t),
                f'y_{i}': m.history['y_scaled'],
                't': m.history['t'],
                f't_change_{i}': m.changepoints_t,
                f'X_{i}': seasonal_features,
                f'sigmas_{i}': prior_scales,
                f'tau_{i}': m.changepoint_prior_scale,
                f'trend_indicator_{i}': trend_indicator[m.growth],
                f's_a_{i}': component_cols['additive_terms'],
                f's_m_{i}': component_cols['multiplicative_terms'],
                f'a_{i}': np.array([1,1,1,1]),
                f'm_{i}': np.array([0,0,0,0]),
                f'X_pred_{i}': seasonal_features_future[df.shape[0]:],
                f'S_pred_{i}': 3
            }
            if m.growth == 'linear':
              dat[f'cap_{i}'] = np.zeros(m.history.shape[0])
              dat[f'cap_pred_{i}'] = np.zeros(only_future.shape[0])
              kinit = m.linear_growth_init(history)
            elif m.growth == 'flat':
              dat[f'cap_{i}'] = np.zeros(m.history.shape[0])
              dat[f'cap_pred_{i}'] = np.zeros(only_future.shape[0])
              kinit = m.flat_growth_init(history)
            else:
              dat[f'cap_{i}'] = history['cap_scaled']
              dat[f'cap_pred_{i}'] = only_future['cap_scaled']
              kinit = m.logistic_growth_init(history)
            stan_init = {
               f'k_{i}': kinit[0],
               f'offset_{i}': kinit[1],
               f'delta_{i}': np.zeros(len(m.changepoints_t)),
               f'beta_{i}': np.zeros(seasonal_features.shape[1]+len(parents)),
               f'sigma_obs_{i}': 1,
               }
            all_dat.update(dat)
            all_init.update(stan_init)
        all_dat['n_samp'] = self.n_samp
        self.dat = all_dat
        model_code = self.generate_stan_code(graph)
        model = pystan.StanModel(model_code=model_code)
        fit = model.optimizing(data=all_dat, init=lambda: all_init, iter=1e4)
        # Loop through nodes again to put forcast results back
        for i in graph.nodes:
            m = graph.nodes[i]["prophet"]
            scale = m.y_scale
            graph.nodes[i]["y_samples"] = fit[f"y_pred_{i}"] * scale
            graph.nodes[i]["y_hat"] = fit[f"y_hat_{i}"] * scale
        return(graph)
