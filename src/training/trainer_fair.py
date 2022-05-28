import os
import json
import warnings
from pathlib import Path

from src.preprocessing import TabPreprocessor
import pandas as pd
from src.FairnessConstraints import DemographicParityLoss




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from scipy.sparse import csc_matrix
from torchmetrics import Metric as TorchMetric
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.losses import ZILNLoss
from src.metrics import Metric, MultipleMetrics
from src.wdtypes import *  # noqa: F403
from src.callbacks import (
    History,
    Callback,
    MetricCallback,
    CallbackContainer,
    LRShedulerCallback,
)
from src.dataloaders import DataLoaderDefault
from src.initializers import Initializer, MultipleInitializer
from src.training._finetune import FineTune
from src.utils.general_utils import Alias
from src.models.tabnet._utils import create_explain_matrix
from src.training._wd_dataset import DatasetObject
from src.training._trainer_utils import (
    alias_to_loss,
    save_epoch_logs,
    wd_train_val_split,
    print_loss_and_metric,
)
from src.training._multiple_optimizer import MultipleOptimizer
from src.training._multiple_transforms import MultipleTransforms
from src.training._loss_and_obj_aliases import _ObjectiveToMethod
from src.training._multiple_lr_scheduler import (
    MultipleLRScheduler,
)

n_cpus = os.cpu_count()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Trainer_fair:
    @Alias(  # noqa: C901
        "objective",
        ["loss_function", "loss_fn", "loss", "cost_function", "cost_fn", "cost"],
    )
    def __init__(
        self,
        model: TabModel,
        objective: str,
        custom_loss_function: Optional[Module] = None,
        optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
        lr_schedulers: Optional[Union[LRScheduler, Dict[str, LRScheduler]]] = None,
        reducelronplateau_criterion: Optional[str] = "loss",
        initializers: Optional[Union[Initializer, Dict[str, Initializer]]] = None,
        transforms: Optional[List[Transforms]] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Union[List[Metric], List[TorchMetric]]] = None,
        tab_preprocessor:  Optional[Module] = None,
        class_weight: Optional[Union[float, List[float], Tuple[float]]] = None,
        lambda_sparse: float = 1e-3,
        alpha: float = 0.25,
        gamma: float = 2,
        verbose: int = 1,
        seed: int = 1,
    ):
        if isinstance(optimizers, Dict):
            if lr_schedulers is not None and not isinstance(lr_schedulers, Dict):
                raise ValueError(
                    "''optimizers' and 'lr_schedulers' must have consistent type: "
                    "(Optimizer and LRScheduler) or (Dict[str, Optimizer] and Dict[str, LRScheduler]) "
                    "Please, read the documentation or see the examples for more details"
                )

        if custom_loss_function is not None and objective not in [
            "binary",
            "multiclass",
            "regression",
        ]:
            raise ValueError(
                "If 'custom_loss_function' is not None, 'objective' must be 'binary' "
                "'multiclass' or 'regression', consistent with the loss function"
            )

        self.reducelronplateau = False
        self.reducelronplateau_criterion = reducelronplateau_criterion
        if isinstance(lr_schedulers, Dict):
            for _, scheduler in lr_schedulers.items():
                if isinstance(scheduler, ReduceLROnPlateau):
                    self.reducelronplateau = True
        elif isinstance(lr_schedulers, ReduceLROnPlateau):
            self.reducelronplateau = True

        self.model = model
        self.tab_preprocessor = tab_preprocessor

        # Tabnet related set ups
        if self.model.is_tabnet:
            self.lambda_sparse = lambda_sparse
            self.reducing_matrix = create_explain_matrix(self.model)

        self.verbose = verbose
        self.seed = seed
        self.objective = objective
        self.method = _ObjectiveToMethod.get(objective)

        # initialize early_stop. If EarlyStopping Callback is used it will
        # take care of it
        self.early_stop = False

        self.loss_fn = self._set_loss_fn(
            objective, class_weight, custom_loss_function, alpha, gamma
        )
        self._initialize(initializers)
        self.optimizer = self._set_optimizer(optimizers)
        self.lr_scheduler = self._set_lr_scheduler(lr_schedulers)
        self.transforms = self._set_transforms(transforms)
        self._set_callbacks_and_metrics(callbacks, metrics)

        self.model.to(device)

    @Alias("finetune", "warmup")  # noqa: C901
    @Alias("finetune_epochs", "warmup_epochs")
    @Alias("finetune_max_lr", "warmup_max_lr")

    @Alias("finetune_deeptabular_gradual", "warmup_deeptabular_gradual")
    @Alias("finetune_deeptabular_max_lr", "warmup_deeptabular_max_lr")
    @Alias("finetune_deeptabular_layers", "warmup_deeptabular_layers")
    @Alias("finetune_routine", "warmup_routine")
    def fit(  # noqa: C901
        self,
        X_tab: Optional[np.ndarray] = None,
        X_train: Optional[Dict[str, np.ndarray]] = None,
        X_val: Optional[Dict[str, np.ndarray]] = None,
        val_split: Optional[float] = None,
        target: Optional[np.ndarray] = None,
        n_epochs: int = 1,
        validation_freq: int = 1,
        batch_size: int = 32,
        custom_dataloader: Union[DataLoader, None] = None,
        finetune: bool = False,
        finetune_epochs: int = 5,
        finetune_max_lr: float = 0.01,
        finetune_deeptabular_gradual: bool = False,
        finetune_deeptabular_max_lr: float = 0.01,
        finetune_deeptabular_layers: Optional[List[nn.Module]] = None,
        finetune_routine: str = "howard",
        stop_after_finetuning: bool = False,
        **kwargs,
    ):
   

        self.batch_size = batch_size
        train_set, eval_set = wd_train_val_split(
            self.seed,
            self.method,
            X_tab,
            X_train,
            X_val,
            val_split,
            target,
        )
        if isinstance(custom_dataloader, type):
            if issubclass(custom_dataloader, DataLoader):
                train_loader = custom_dataloader(
                    dataset=train_set,
                    batch_size=batch_size,
                    num_workers=n_cpus,
                    **kwargs,
                )
            else:
                NotImplementedError(
                    "Custom DataLoader must be a subclass of "
                    "torch.utils.data.DataLoader, please see the "
                    "pytorch documentation or examples in "
                    "src.dataloaders"
                )
        else:
            train_loader = DataLoaderDefault(
                dataset=train_set, batch_size=batch_size, num_workers=n_cpus
            )
        train_steps = len(train_loader)
        if eval_set is not None:
            eval_loader = DataLoader(
                dataset=eval_set,
                batch_size=batch_size,
                num_workers=n_cpus,
                shuffle=False,
            )
            eval_steps = len(eval_loader)

        if finetune:
            self._finetune(
                train_loader,
                finetune_epochs,
                finetune_max_lr,
                finetune_deeptabular_gradual,
                finetune_deeptabular_layers,
                finetune_deeptabular_max_lr,
                finetune_routine,
            )
            if stop_after_finetuning:
                print("Fine-tuning finished")
                return
            else:
                if self.verbose:
                    print(
                        "Fine-tuning of individual components completed. "
                        "Training the whole model for {} epochs".format(n_epochs)
                    )

        self.callback_container.on_train_begin(
            {"batch_size": batch_size, "train_steps": train_steps, "n_epochs": n_epochs}
        )
        for epoch in range(n_epochs):
            epoch_logs: Dict[str, float] = {}
            self.callback_container.on_epoch_begin(epoch, logs=epoch_logs)

            self.train_running_loss = 0.0
            with trange(train_steps, disable=self.verbose != 1) as t:
                for batch_idx, (data, targett) in zip(t, train_loader):
                    t.set_description("epoch %i" % (epoch + 1))
                    train_score, train_loss = self._train_step(data, targett, batch_idx, self.tab_preprocessor)
                    print_loss_and_metric(t, train_loss, train_score)
                    self.callback_container.on_batch_end(batch=batch_idx)
            epoch_logs = save_epoch_logs(epoch_logs, train_loss, train_score, "train")

            on_epoch_end_metric = None
            if eval_set is not None and epoch % validation_freq == (
                validation_freq - 1
            ):
                self.callback_container.on_eval_begin()
                self.valid_running_loss = 0.0
                with trange(eval_steps, disable=self.verbose != 1) as v:
                    for i, (data, targett) in zip(v, eval_loader):
                        v.set_description("valid")
                        val_score, val_loss = self._eval_step(data, targett, i)
                        print_loss_and_metric(v, val_loss, val_score)
                epoch_logs = save_epoch_logs(epoch_logs, val_loss, val_score, "val")

                if self.reducelronplateau:
                    if self.reducelronplateau_criterion == "loss":
                        on_epoch_end_metric = val_loss
                    else:
                        on_epoch_end_metric = val_score[
                            self.reducelronplateau_criterion
                        ]

            self.callback_container.on_epoch_end(epoch, epoch_logs, on_epoch_end_metric)

            if self.early_stop:
                self.callback_container.on_train_end(epoch_logs)
                break

        self.callback_container.on_train_end(epoch_logs)
        if self.model.is_tabnet:
            self._compute_feature_importance(train_loader)
        self._restore_best_weights()
        self.model.train()

    def predict(  # type: ignore[return]
        self,
        X_tab: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
    ) -> np.ndarray:

        preds_l = self._predict(X_tab, X_test, batch_size)
        if self.method == "regression":
            return np.vstack(preds_l).squeeze(1)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            return (preds > 0.5).astype("int")
        if self.method == "qregression":
            return np.vstack(preds_l)
        if self.method == "multiclass":
            preds = np.vstack(preds_l)
            return np.argmax(preds, 1)  # type: ignore[return-value]

    def _predict_ziln(self, preds: Tensor) -> Tensor:
        """Calculates predicted mean of zero inflated lognormal logits.
        Adjusted implementaion of `code
        <https://github.com/google/lifetime_value/blob/master/lifetime_value/zero_inflated_lognormal.py>`
        Arguments:
            preds: [batch_size, 3] tensor of logits.
        Returns:
            ziln_preds: [batch_size, 1] tensor of predicted mean.
        """
        positive_probs = torch.sigmoid(preds[..., :1])
        loc = preds[..., 1:2]
        scale = F.softplus(preds[..., 2:])
        ziln_preds = positive_probs * torch.exp(loc + 0.5 * torch.square(scale))
        return ziln_preds

    def predict_uncertainty(  # type: ignore[return]
        self,
        X_tab: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
        uncertainty_granularity=1000,
    ) -> np.ndarray:
        r"""Returns the predicted ucnertainty of the model for the test dataset using a
        Monte Carlo method during which dropout layers are activated in the evaluation/prediction
        phase and each sample is predicted N times (uncertainty_granularity times). Based on [1].

        [1] Gal Y. & Ghahramani Z., 2016, Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning, Proceedings of the 33rd International Conference on Machine Learning

        """
        preds_l = self._predict(
            X_tab,
            X_test,
            batch_size,
            uncertainty_granularity,
            uncertainty=True,
        )
        preds = np.vstack(preds_l)
        samples_num = int(preds.shape[0] / uncertainty_granularity)
        if self.method == "regression":
            preds = preds.squeeze(1)
            preds = preds.reshape((uncertainty_granularity, samples_num))
            return np.array(
                (
                    preds.max(axis=0),
                    preds.min(axis=0),
                    preds.mean(axis=0),
                    preds.std(axis=0),
                )
            ).T
        if self.method == "qregression":
            raise ValueError(
                "Currently predict_uncertainty is not supported for qregression method"
            )
        if self.method == "binary":
            preds = preds.squeeze(1)
            preds = preds.reshape((uncertainty_granularity, samples_num))
            preds = preds.mean(axis=0)
            probs = np.zeros([preds.shape[0], 3])
            probs[:, 0] = 1 - preds
            probs[:, 1] = preds
            return probs
        if self.method == "multiclass":
            preds = preds.reshape(uncertainty_granularity, samples_num, preds.shape[1])
            preds = preds.mean(axis=0)
            preds = np.hstack((preds, np.vstack(np.argmax(preds, 1))))
            return preds

    def predict_proba(  # type: ignore[return]
        self,
        X_tab: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
    ) -> np.ndarray:
    

        preds_l = self._predict(X_tab, X_test, batch_size)
        if self.method == "binary":
            preds = np.vstack(preds_l).squeeze(1)
            probs = np.zeros([preds.shape[0], 2])
            probs[:, 0] = 1 - preds
            probs[:, 1] = preds
            return probs
        if self.method == "multiclass":
            return np.vstack(preds_l)

    def get_embeddings(
        self, col_name: str, cat_encoding_dict: Dict[str, Dict[str, int]]
    ) -> Dict[str, np.ndarray]:  # pragma: no cover
    
        warnings.warn(
            "'get_embeddings' will be deprecated in the next release. "
            "Please consider using 'Tab2vec' instead",
            DeprecationWarning,
        )
        for n, p in self.model.named_parameters():
            if "embed_layers" in n and col_name in n:
                embed_mtx = p.cpu().data.numpy()
        encoding_dict = cat_encoding_dict[col_name]
        inv_encoding_dict = {v: k for k, v in encoding_dict.items()}
        cat_embed_dict = {}
        for idx, value in inv_encoding_dict.items():
            cat_embed_dict[value] = embed_mtx[idx]
        return cat_embed_dict

    def explain(self, X_tab: np.ndarray, save_step_masks: bool = False):
   
        loader = DataLoader(
            dataset=DatasetObject(**{"X_tab": X_tab}),
            batch_size=self.batch_size,
            num_workers=n_cpus,
            shuffle=False,
        )

        self.model.eval()
        tabnet_backbone = list(self.model.deeptabular.children())[0]

        m_explain_l = []
        for batch_nb, data in enumerate(loader):
            X = data["deeptabular"].to(device)
            M_explain, masks = tabnet_backbone.forward_masks(X)  # type: ignore[operator]
            m_explain_l.append(
                csc_matrix.dot(M_explain.cpu().detach().numpy(), self.reducing_matrix)
            )
            if save_step_masks:
                for key, value in masks.items():
                    masks[key] = csc_matrix.dot(
                        value.cpu().detach().numpy(), self.reducing_matrix
                    )
                if batch_nb == 0:
                    m_explain_step = masks
                else:
                    for key, value in masks.items():
                        m_explain_step[key] = np.vstack([m_explain_step[key], value])

        m_explain_agg = np.vstack(m_explain_l)
        m_explain_agg_norm = m_explain_agg / m_explain_agg.sum(axis=1)[:, np.newaxis]

        res = (
            (m_explain_agg_norm, m_explain_step)
            if save_step_masks
            else np.vstack(m_explain_agg_norm)
        )

        return res

    def save(
        self,
        path: str,
        save_state_dict: bool = False,
        model_filename: str = "wd_model.pt",
    ):
  

        save_dir = Path(path)
        history_dir = save_dir / "history"
        history_dir.mkdir(exist_ok=True, parents=True)

        # the trainer is run with the History Callback by default
        with open(history_dir / "train_eval_history.json", "w") as teh:
            json.dump(self.history, teh)  # type: ignore[attr-defined]

        has_lr_history = any(
            [clbk.__class__.__name__ == "LRHistory" for clbk in self.callbacks]
        )
        if self.lr_scheduler is not None and has_lr_history:
            with open(history_dir / "lr_history.json", "w") as lrh:
                json.dump(self.lr_history, lrh)  # type: ignore[attr-defined]

        model_path = save_dir / model_filename
        if save_state_dict:
            torch.save(self.model.state_dict(), model_path)
        else:
            torch.save(self.model, model_path)

        if self.model.is_tabnet:
            with open(save_dir / "feature_importance.json", "w") as fi:
                json.dump(self.feature_importance, fi)

    def _restore_best_weights(self):
        already_restored = any(
            [
                (
                    callback.__class__.__name__ == "EarlyStopping"
                    and callback.restore_best_weights
                )
                for callback in self.callback_container.callbacks
            ]
        )
        if already_restored:
            pass
        else:
            for callback in self.callback_container.callbacks:
                if callback.__class__.__name__ == "ModelCheckpoint":
                    if callback.save_best_only:
                        if self.verbose:
                            print(
                                f"Model weights restored to best epoch: {callback.best_epoch + 1}"
                            )
                        self.model.load_state_dict(callback.best_state_dict)
                    else:
                        if self.verbose:
                            print(
                                "Model weights after training corresponds to the those of the "
                                "final epoch which might not be the best performing weights. Use"
                                "the 'ModelCheckpoint' Callback to restore the best epoch weights."
                            )

    def _finetune(
        self,
        loader: DataLoader,
        n_epochs: int,
        max_lr: float,
        deeptabular_gradual: bool,
        deeptabular_layers: List[nn.Module],
        deeptabular_max_lr: float,
        routine: str = "felbo",
    ):  # pragma: no cover
        r"""
        Simple wrap-up to individually fine-tune model components
        """
        if self.model.deephead is not None:
            raise ValueError(
                "Currently warming up is only supported without a fully connected 'DeepHead'"
            )
        # This is not the most elegant solution, but is a soluton "in-between"
        # a non elegant one and re-factoring the whole code
        finetuner = FineTune(self.loss_fn, self.metric, self.method, self.verbose)
        if self.model.wide:
            finetuner.finetune_all(self.model.wide, "wide", loader, n_epochs, max_lr)
        if self.model.deeptabular:
            if deeptabular_gradual:
                finetuner.finetune_gradual(
                    self.model.deeptabular,
                    "deeptabular",
                    loader,
                    deeptabular_max_lr,
                    deeptabular_layers,
                    routine,
                )
            else:
                finetuner.finetune_all(
                    self.model.deeptabular, "deeptabular", loader, n_epochs, max_lr
                )


    def _train_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int, tab_preprocessor):
        self.model.train()
        X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
        y = (
            target.view(-1, 1).float()
            if self.method not in ["multiclass", "qregression"]
            else target
        )
        y = y.to(device)

        self.optimizer.zero_grad()
        y_pred = self.model(X)

        # Prediction
        pred_train_list_batch = []
        list_train_pred_batch = []
        list_train_pred_batch.append(torch.sigmoid(y_pred).round()) 
        for i in range(len(list_train_pred_batch)):
                for j in range(len(list_train_pred_batch[i])):
                    pred_train_list_batch.append(list_train_pred_batch[i][j][0].tolist())
        df_pred_batch = pd.DataFrame(pred_train_list_batch, columns = ['df_pred'])
        #print('df_pred_batch', df_pred_batch.shape)


        # Taregt
        target_train_list_batch = []
        list_all_train_target_batch = []
        list_all_train_target_batch.append(y)
        for i in range(len(list_all_train_target_batch)):
                for j in range(len(list_all_train_target_batch[i])):
                    target_train_list_batch.append(list_all_train_target_batch[i][j].tolist()[0])
        df_target_batch = pd.DataFrame(target_train_list_batch, columns = ['df_target'])
        #print('df_target_batch', df_target_batch.shape)


        # Input
        both_cont_and_cat_col = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime','fam_inc' ,'male', 'race', 'tier']

        list_all_train_data_batch = []
        train_data_list_batch = []

        list_all_train_data_batch.append(X)    #  list_all_train_data_batch
        df_input_batch = pd.DataFrame(columns = both_cont_and_cat_col)

        for i in range(len(list_all_train_data_batch)):
            for j in range(len(list_all_train_data_batch[i]['deeptabular'])):
                train_data_list_batch.append(np.asarray([list_all_train_data_batch[i]['deeptabular'][j].tolist()]))

        for i in range(len(train_data_list_batch)):
            row_batch = tab_preprocessor.inverse_transform(train_data_list_batch[i])
            df_input_batch = pd.concat([df_input_batch, row_batch], ignore_index=True)

        #print('df_input_batch', df_input_batch.shape)
        Final_DataFrame_batch = pd.concat([df_input_batch, df_target_batch, df_pred_batch], axis=1)
        
        # Fairness
        Final_DataFrame_batch.loc[:, "race"] = Final_DataFrame_batch.loc[:, "race"].astype(str)
        Final_DataFrame_batch.loc[:, "race"] = Final_DataFrame_batch.loc[:, "race"].where(Final_DataFrame_batch.loc[:, "race"] == 'White', 0)
        Final_DataFrame_batch.loc[:, "race"] = Final_DataFrame_batch.loc[:, "race"].where(Final_DataFrame_batch.loc[:, "race"] != 'White', 1)
        Final_DataFrame_batch.loc[:, "race"] = Final_DataFrame_batch.loc[:, "race"].astype(int)
        sensitive_feature_elements="race"
        sensitive_feature_train = Final_DataFrame_batch.loc[:, sensitive_feature_elements]
        sensitive_feature_train = torch.from_numpy(sensitive_feature_train.values).float()

        fairness_constraint = DemographicParityLoss(alpha=100,
                                             sensitive_classes=Final_DataFrame_batch[sensitive_feature_elements].unique().astype(int).tolist(),
                                              )
        sensitive_feature=sensitive_feature_train
        penalty = fairness_constraint(X, y_pred.view(-1), sensitive_feature, y)

        ################   
        
        if self.model.is_tabnet:
            loss = self.loss_fn(y_pred[0], y) - self.lambda_sparse * y_pred[1]
            score = self._get_score(y_pred[0], y)
        else:
            loss = self.loss_fn(y_pred, y)
            score = self._get_score(y_pred, y)
        # TODO raise exception if the loss is exploding with non scaled target values

        loss = loss + penalty
        loss.backward()
        self.optimizer.step()

        self.train_running_loss += loss.item()
        avg_loss = self.train_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _eval_step(self, data: Dict[str, Tensor], target: Tensor, batch_idx: int):

        self.model.eval()
        with torch.no_grad():
            X = {k: v.cuda() for k, v in data.items()} if use_cuda else data
            y = (
                target.view(-1, 1).float()
                if self.method not in ["multiclass", "qregression"]
                else target
            )
            y = y.to(device)

            y_pred = self.model(X)
            if self.model.is_tabnet:
                loss = self.loss_fn(y_pred[0], y) - self.lambda_sparse * y_pred[1]
                score = self._get_score(y_pred[0], y)
            else:
                score = self._get_score(y_pred, y)
                loss = self.loss_fn(y_pred, y)

            self.valid_running_loss += loss.item()
            avg_loss = self.valid_running_loss / (batch_idx + 1)

        return score, avg_loss

    def _get_score(self, y_pred, y):
        if self.metric is not None:
            if self.method == "regression":
                score = self.metric(y_pred, y)
            if self.method == "binary":
                score = self.metric(torch.sigmoid(y_pred), y)
            if self.method == "qregression":
                score = self.metric(y_pred, y)
            if self.method == "multiclass":
                score = self.metric(F.softmax(y_pred, dim=1), y)
            return score
        else:
            return None

    def _compute_feature_importance(self, loader: DataLoader):
        self.model.eval()
        tabnet_backbone = list(self.model.deeptabular.children())[0]
        feat_imp = np.zeros((tabnet_backbone.embed_and_cont_dim))  # type: ignore[arg-type]
        for data, target in loader:
            X = data["deeptabular"].to(device)
            y = target.view(-1, 1).float() if self.method != "multiclass" else target
            y = y.to(device)
            M_explain, masks = tabnet_backbone.forward_masks(X)  # type: ignore[operator]
            feat_imp += M_explain.sum(dim=0).cpu().detach().numpy()

        feat_imp = csc_matrix.dot(feat_imp, self.reducing_matrix)
        feat_imp = feat_imp / np.sum(feat_imp)

        self.feature_importance = {
            k: v for k, v in zip(tabnet_backbone.column_idx.keys(), feat_imp)  # type: ignore[operator, union-attr]
        }

    def _predict(  # noqa: C901
        self,
        X_tab: Optional[np.ndarray] = None,
        X_test: Optional[Dict[str, np.ndarray]] = None,
        batch_size: int = 256,
        uncertainty_granularity=1000,
        uncertainty: bool = False,
        quantiles: bool = False,
    ) -> List:
        r"""Private method to avoid code repetition in predict and
        predict_proba. For parameter information, please, see the .predict()
        method documentation
        """
        if X_test is not None:
            test_set = DatasetObject(**X_test)

        if not hasattr(self, "batch_size"):
            self.batch_size = batch_size

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            num_workers=n_cpus,
            shuffle=False,
        )
        test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1  # type: ignore[arg-type]

        self.model.eval()
        preds_l = []

        if uncertainty:
            for m in self.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
            prediction_iters = uncertainty_granularity
        else:
            prediction_iters = 1

        with torch.no_grad():
            with trange(uncertainty_granularity, disable=uncertainty is False) as t:
                for i, k in zip(t, range(prediction_iters)):
                    t.set_description("predict_UncertaintyIter")

                    with trange(
                        test_steps, disable=self.verbose != 1 or uncertainty is True
                    ) as tt:
                        for j, data in zip(tt, test_loader):
                            tt.set_description("predict")
                            X = (
                                {k: v.cuda() for k, v in data.items()}
                                if use_cuda
                                else data
                            )
                            preds = (
                                self.model(X)
                                if not self.model.is_tabnet
                                else self.model(X)[0]
                            )
                            if self.method == "binary":
                                preds = torch.sigmoid(preds)
                            if self.method == "multiclass":
                                preds = F.softmax(preds, dim=1)
                            if self.method == "regression" and isinstance(
                                self.loss_fn, ZILNLoss
                            ):
                                preds = self._predict_ziln(preds)
                            preds = preds.cpu().data.numpy()
                            preds_l.append(preds)
        self.model.train()
        return preds_l

    def _set_loss_fn(self, objective, class_weight, custom_loss_function, alpha, gamma):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight).to(device)
        if custom_loss_function is not None:
            return custom_loss_function
        elif (
            self.method not in ["regression", "qregression"]
            and "focal_loss" not in objective
        ):
            return alias_to_loss(objective, weight=class_weight)
        elif "focal_loss" in objective:
            return alias_to_loss(objective, alpha=alpha, gamma=gamma)
        else:
            return alias_to_loss(objective)

    def _initialize(self, initializers):
        if initializers is not None:
            if isinstance(initializers, Dict):
                self.initializer = MultipleInitializer(
                    initializers, verbose=self.verbose
                )
                self.initializer.apply(self.model)
            elif isinstance(initializers, type):
                self.initializer = initializers()
                self.initializer(self.model)
            elif isinstance(initializers, Initializer):
                self.initializer = initializers
                self.initializer(self.model)

    def _set_optimizer(self, optimizers):
        if optimizers is not None:
            if isinstance(optimizers, Optimizer):
                optimizer: Union[Optimizer, MultipleOptimizer] = optimizers
            elif isinstance(optimizers, Dict):
                opt_names = list(optimizers.keys())
                mod_names = [n for n, c in self.model.named_children()]
                for mn in mod_names:
                    assert mn in opt_names, "No optimizer found for {}".format(mn)
                optimizer = MultipleOptimizer(optimizers)
        else:
            optimizer = torch.optim.Adam(self.model.parameters())  # type: ignore
        return optimizer

    def _set_lr_scheduler(self, lr_schedulers):
        if lr_schedulers is not None:
            # ReduceLROnPlateau is special, only scheduler that is 'just' an
            # object rather than a LRScheduler
            if isinstance(lr_schedulers, LRScheduler) or isinstance(
                lr_schedulers, ReduceLROnPlateau
            ):
                lr_scheduler = lr_schedulers
                cyclic_lr = "cycl" in lr_scheduler.__class__.__name__.lower()
            else:
                lr_scheduler = MultipleLRScheduler(lr_schedulers)
                scheduler_names = [
                    sc.__class__.__name__.lower()
                    for _, sc in lr_scheduler._schedulers.items()
                ]
                cyclic_lr = any(["cycl" in sn for sn in scheduler_names])
        else:
            lr_scheduler, cyclic_lr = None, False
        self.cyclic_lr = cyclic_lr
        return lr_scheduler

    @staticmethod
    def _set_transforms(transforms):
        if transforms is not None:
            return MultipleTransforms(transforms)()
        else:
            return None

    def _set_callbacks_and_metrics(self, callbacks, metrics):
        self.callbacks: List = [History(), LRShedulerCallback()]
        if callbacks is not None:
            for callback in callbacks:
                if isinstance(callback, type):
                    callback = callback()
                self.callbacks.append(callback)
        if metrics is not None:
            self.metric = MultipleMetrics(metrics)
            self.callbacks += [MetricCallback(self.metric)]
        else:
            self.metric = None
        self.callback_container = CallbackContainer(self.callbacks)
        self.callback_container.set_model(self.model)
        self.callback_container.set_trainer(self)
