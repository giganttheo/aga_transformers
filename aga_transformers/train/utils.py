import jax

from ..models.utils import add_graph_to_params

def get_apply_fn(model):
    """
    this function can be used in flax.training.train_state.TrainState as an apply_fn instead of model.__call__
    otherwise the optimizer will not accept the graph with a different dtype as the params (even if we ignore the graph in the optimizer)
    """
    apply_fn_with_graph = lambda params, graph, **kwargs: model.__call__(params=add_graph_to_params(params, graph), **kwargs)
    return apply_fn_with_graph
