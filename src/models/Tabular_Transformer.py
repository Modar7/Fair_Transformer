
import warnings

import torch
import torch.nn as nn

from src.wdtypes import *  # noqa: F403
from src.models.tab_mlp import MLP, get_activation_fn
from src.models.tabnet.tab_net import TabNetPredLayer

warnings.filterwarnings("default", category=UserWarning)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class TabModel(nn.Module):
    def __init__(
        self,
        wide: Optional[nn.Module] = None,
        deeptabular: Optional[nn.Module] = None,
        deephead: Optional[nn.Module] = None,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: float = 0.1,
        head_batchnorm: bool = False,
        head_batchnorm_last: bool = False,
        head_linear_first: bool = False,
        enforce_positive: bool = False,
        enforce_positive_activation: str = "softplus",
        pred_dim: int = 1,
    ):
        super(TabModel, self).__init__()

        self._check_model_components(
            deeptabular,
            deephead,
            head_hidden_dims,
            pred_dim,
        )

        # required as attribute just in case we pass a deephead
        self.pred_dim = pred_dim

        # The main components
        self.wide = wide
        self.deeptabular = deeptabular
        self.deephead = deephead
        self.enforce_positive = enforce_positive

        if self.deeptabular is not None:
            self.is_tabnet = deeptabular.__class__.__name__ == "TabNet"
        else:
            self.is_tabnet = False

        if self.deephead is None:
            if head_hidden_dims is not None:
                self._build_deephead(
                    head_hidden_dims,
                    head_activation,
                    head_dropout,
                    head_batchnorm,
                    head_batchnorm_last,
                    head_linear_first,
                )
            else:
                self._add_pred_layer()

        if self.enforce_positive:
            self.enf_pos = get_activation_fn(enforce_positive_activation)

    def forward(self, X: Dict[str, Tensor]):
        wide_out = self._forward_wide(X)
        if self.deephead:
            deep = self._forward_deephead(X, wide_out)
        else:
            deep = self._forward_deep(X, wide_out)
        if self.enforce_positive:
            return self.enf_pos(deep)
        else:
            return deep

    def _build_deephead(
        self,
        head_hidden_dims,
        head_activation,
        head_dropout,
        head_batchnorm,
        head_batchnorm_last,
        head_linear_first,
    ):
        deep_dim = 0
        if self.deeptabular is not None:
            deep_dim += self.deeptabular.output_dim

        head_hidden_dims = [deep_dim] + head_hidden_dims
        self.deephead = MLP(
            head_hidden_dims,
            head_activation,
            head_dropout,
            head_batchnorm,
            head_batchnorm_last,
            head_linear_first,
        )
        self.deephead.add_module(
            "head_out", nn.Linear(head_hidden_dims[-1], self.pred_dim)
        )

    def _add_pred_layer(self):
        if self.deeptabular is not None:
            if self.is_tabnet:
                self.deeptabular = nn.Sequential(
                    self.deeptabular,
                    TabNetPredLayer(self.deeptabular.output_dim, self.pred_dim),
                )
            else:
                self.deeptabular = nn.Sequential(
                    self.deeptabular,
                    nn.Linear(self.deeptabular.output_dim, self.pred_dim),
                )

    def _forward_wide(self, X):
        if self.wide is not None:
            out = self.wide(X["wide"])
        else:
            batch_size = X[list(X.keys())[0]].size(0)
            out = torch.zeros(batch_size, self.pred_dim).to(device)

        return out

    def _forward_deephead(self, X, wide_out):
        if self.deeptabular is not None:
            if self.is_tabnet:
                tab_out = self.deeptabular(X["deeptabular"])
                deepside, M_loss = tab_out[0], tab_out[1]
            else:
                deepside = self.deeptabular(X["deeptabular"])
        else:
            deepside = torch.FloatTensor().to(device)

        deephead_out = self.deephead(deepside)
        deepside_out = nn.Linear(deephead_out.size(1), self.pred_dim).to(device)

        if self.is_tabnet:
            res = (wide_out.add_(deepside_out(deephead_out)), M_loss)
        else:
            res = wide_out.add_(deepside_out(deephead_out))

        return res

    def _forward_deep(self, X, wide_out):
        if self.deeptabular is not None:
            if self.is_tabnet:
                tab_out, M_loss = self.deeptabular(X["deeptabular"])
                wide_out.add_(tab_out)
            else:
                wide_out.add_(self.deeptabular(X["deeptabular"]))

        if self.is_tabnet:
            res = (wide_out, M_loss)
        else:
            res = wide_out

        return res

    @staticmethod  # noqa: C901
    def _check_model_components(  # noqa: C901
        deeptabular,
        deephead,
        head_hidden_dims,
        pred_dim,
    ):
        if deeptabular is not None and not hasattr(deeptabular, "output_dim"):
            raise AttributeError(
                "deeptabular model must have an 'output_dim' attribute. "
                "See src.models.deep_text.DeepText"
            )
        if deeptabular is not None:
            is_tabnet = deeptabular.__class__.__name__ == "TabNet"
        if deephead is not None and head_hidden_dims is not None:
            raise ValueError(
                "both 'deephead' and 'head_hidden_dims' are not None. Use one of the other, but not both"
            )
        if (
            head_hidden_dims is not None
            and not deeptabular
        ):
            raise ValueError(
                "if 'head_hidden_dims' is not None, at least one deep component must be used"
            )
        if deephead is not None:
            deephead_inp_feat = next(deephead.parameters()).size(1)
            output_dim = 0
            if deeptabular is not None:
                output_dim += deeptabular.output_dim
            assert deephead_inp_feat == output_dim, (
                "if a custom 'deephead' is used its input features ({}) must be equal to "
                "the output features of the deep component ({})".format(
                    deephead_inp_feat, output_dim
                )
            )
