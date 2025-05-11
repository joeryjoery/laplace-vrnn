from typing import Sequence

import jax
import jax.numpy as jnp

from lvrnn import distributions as dist

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_PLOT_HEIGHT: int = 500
DEFAULT_PLOT_WIDTH: int = 400


def plot_2D_function_predictive(
        distributions: dist.SerializeTree[dist.MultivariateNormalTriangular],
        observations: tuple[jax.Array, jax.Array],
        true_data: tuple[jax.Array, jax.Array],
        data_sizes: Sequence[int],
        scatter: bool
) -> go.Figure:
    raise NotImplementedError("TODO plot2d")


def plot_1D_function_predictive(
        distributions: dist.SerializeTree[dist.MultivariateNormalTriangular],
        observations: tuple[jax.Array, jax.Array],
        true_data: tuple[jax.Array, jax.Array],
        data_sizes: Sequence[int]
) -> go.Figure:

    # Sort scatter data-domain for line-plotting.
    x, y = true_data
    idx = x.argsort()

    n = len(jax.tree_util.tree_leaves(distributions)[0])
    fig = make_subplots(
        rows=1, cols=n,
        horizontal_spacing=0.02,
        subplot_titles=[
            f'Sample-Size = {m}' for m in data_sizes
        ],
    )
    for j, m in enumerate(data_sizes):
        y_hat_cond_action = jax.tree_map(lambda arr: arr[j], distributions)

        mu, var = jax.vmap(lambda d: (d.get.mean(), d.get.variance()))(
            y_hat_cond_action
        )
        sigma = jnp.sqrt(var + 1e-8)

        unwrapped, _ = y_hat_cond_action.vargs  # Extract out of ensemble
        sample_means = jax.vmap(lambda d: d.get.mean())(unwrapped)

        mu, sigma, sample_means = jax.tree_map(
            lambda arr: jnp.squeeze(arr, axis=-1), (mu, sigma, sample_means)
        )

        plot_data = populate_1D_function_predictive(
            (x.at[idx].get(), y.at[idx].get()),
            (x.at[idx].get(), mu.at[idx].get()),
            (x.at[idx].get(), sigma.at[idx].get()),
            (x.at[idx].get(), sample_means.at[idx].get()),
            (observations[0][:m], observations[1][:m]),
            annotate=(j == 1)  # omit 0th step due ot empty observations.
        )
        _ = [fig.add_trace(data, row=1, col=j+1) for data in plot_data]

    fig.update_layout(
        height=DEFAULT_PLOT_HEIGHT, width=DEFAULT_PLOT_WIDTH*n,  # Plot size
        plot_bgcolor='white',  # Background
        showlegend=True,
        title_text='Model Posterior Predictive', title_x=0.5
    )
    fig.update_yaxes(
        gridcolor='rgba(0, 0, 0, 0.2)',  # Background
        showline=True, linewidth=2, linecolor='black'  # Axes
    )
    fig.update_xaxes(
        gridcolor='rgba(0, 0, 0, 0.2)',  # Background
        showline=True, linewidth=2, linecolor='black'  # Axes
    )

    return fig


def populate_1D_function_predictive(
        true_data: tuple[jax.Array, jax.Array],
        predict_data_mean: tuple[jax.Array, jax.Array],
        predict_data_std: tuple[jax.Array, jax.Array],
        predict_data_samples: tuple[jax.Array, jax.Array],
        observed_data: tuple[jax.Array, jax.Array],
        annotate: bool = False
) -> list[go.Scatter]:

    true = go.Scatter(
        x=true_data[0], y=true_data[1],
        name='True Function', showlegend=annotate,
        mode='lines', line=dict(
            width=2, color=u'#1f77b4'
        )
    )
    mean = go.Scatter(
        x=predict_data_mean[0], y=predict_data_mean[1],
        name='Ensemble Predictive Mean', showlegend=annotate,
        mode='lines', line=dict(width=2, color=u'#ff7f0e')
    )
    plus_stddev = go.Scatter(
        x=predict_data_mean[0], y=predict_data_mean[1] + predict_data_std[1],
        name='Ensemble Predictive Mean +- Stddev', showlegend=annotate,
        mode='lines', line=dict(width=2, dash='dash', color=u'#ff7f0e')
    )
    min_stddev = go.Scatter(
        x=predict_data_mean[0], y=predict_data_mean[1] - predict_data_std[1],
        name='Ensemble Predictive Mean - Stddev', showlegend=False,
        mode='lines', line=dict(width=2, dash='dash', color=u'#ff7f0e')
    )
    observations = go.Scatter(
        x=observed_data[0], y=observed_data[1],
        mode='markers', name='Observations', showlegend=annotate,
        marker=dict(
            size=5, symbol='x', color='black'
        )
    )

    samples = list()
    for i, line in enumerate(predict_data_samples[1].T):
        samples.append(go.Scatter(
            x=predict_data_samples[0], y=line,
            name='Ensemble Member Means',
            showlegend=(annotate and (i == 0)),
            mode='lines', line=dict(
                width=1, color='rgba(0, 150, 0, 0.2)'
            )
        ))

    return samples + [mean, plus_stddev, min_stddev, true, observations]
