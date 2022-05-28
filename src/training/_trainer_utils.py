import numpy as np
from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split

from src.losses import (
    MSLELoss,
    RMSELoss,
    ZILNLoss,
    FocalLoss,
    RMSLELoss,
    TweedieLoss,
    QuantileLoss,
)
from src.wdtypes import Dict, List, Optional, Transforms
from src.training._wd_dataset import DatasetObject
from src.training._loss_and_obj_aliases import (
    _LossAliases,
    _ObjectiveToMethod,
)

def wd_train_val_split(  # noqa: C901
    seed: int,
    method: str,
    X_tab: Optional[np.ndarray] = None,
    X_train: Optional[Dict[str, np.ndarray]] = None,
    X_val: Optional[Dict[str, np.ndarray]] = None,
    val_split: Optional[float] = None,
    target: Optional[np.ndarray] = None,
    transforms: Optional[List[Transforms]] = None,
):

    if X_val is not None:
        assert (
            X_train is not None
        ), "if the validation set is passed as a dictionary, the training set must also be a dictionary"
        train_set = DatasetObject(**X_train, transforms=transforms)  # type: ignore
        eval_set = DatasetObject(**X_val, transforms=transforms)  # type: ignore
    elif val_split is not None:
        if not X_train:
            X_train = _build_train_dict(X_tab, target)
        y_tr, y_val, idx_tr, idx_val = train_test_split(
            X_train["target"],
            np.arange(len(X_train["target"])),
            test_size=val_split,
            random_state=seed,
            stratify=X_train["target"] if method != "regression" else None,
        )
        X_tr, X_val = {"target": y_tr}, {"target": y_val}
        if "X_tab" in X_train.keys():
            X_tr["X_tab"], X_val["X_tab"] = (
                X_train["X_tab"][idx_tr],
                X_train["X_tab"][idx_val],
            )
        train_set = DatasetObject(**X_tr, transforms=transforms)  # type: ignore
        eval_set = DatasetObject(**X_val, transforms=transforms)  # type: ignore
    else:
        if not X_train:
            X_train = _build_train_dict(X_tab, target)
        train_set = DatasetObject(**X_train, transforms=transforms)  # type: ignore
        eval_set = None

    return train_set, eval_set

def _build_train_dict(X_tab, target):
    X_train = {"target": target}
    if X_tab is not None:
        X_train["X_tab"] = X_tab
    return X_train


def print_loss_and_metric(pb: tqdm, loss: float, score: Dict):
    if score is not None:
        pb.set_postfix(
            metrics={
                k: np.round(v.astype(float), 4).tolist() for k, v in score.items()
            },
            loss=loss,
        )
    else:
        pb.set_postfix(loss=loss)


def save_epoch_logs(epoch_logs: Dict, loss: float, score: Dict, stage: str):

    epoch_logs["_".join([stage, "loss"])] = loss
    if score is not None:
        for k, v in score.items():
            log_k = "_".join([stage, k])
            epoch_logs[log_k] = v
    return epoch_logs


def alias_to_loss(loss_fn: str, **kwargs):  # noqa: C901
    if loss_fn not in _ObjectiveToMethod.keys():
        raise ValueError(
            "objective or loss function is not supported. Please consider passing a callable "
            "directly to the compile method (see docs) or use one of the supported objectives "
            "or loss functions: {}".format(", ".join(_ObjectiveToMethod.keys()))
        )
    if loss_fn in _LossAliases.get("binary"):
        return nn.BCEWithLogitsLoss(pos_weight=kwargs["weight"])
    if loss_fn in _LossAliases.get("multiclass"):
        return nn.CrossEntropyLoss(weight=kwargs["weight"])
    if loss_fn in _LossAliases.get("regression"):
        return nn.MSELoss()
    if loss_fn in _LossAliases.get("mean_absolute_error"):
        return nn.L1Loss()
    if loss_fn in _LossAliases.get("mean_squared_log_error"):
        return MSLELoss()
    if loss_fn in _LossAliases.get("root_mean_squared_error"):
        return RMSELoss()
    if loss_fn in _LossAliases.get("root_mean_squared_log_error"):
        return RMSLELoss()
    if loss_fn in _LossAliases.get("zero_inflated_lognormal"):
        return ZILNLoss()
    if loss_fn in _LossAliases.get("quantile"):
        return QuantileLoss()
    if loss_fn in _LossAliases.get("tweedie"):
        return TweedieLoss()
    if "focal_loss" in loss_fn:
        return FocalLoss(**kwargs)
