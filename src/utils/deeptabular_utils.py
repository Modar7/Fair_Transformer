import warnings

import pandas as pd
from sklearn.exceptions import NotFittedError

from ..wdtypes import *  # noqa: F403

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

__all__ = ["LabelEncoder"]


class LabelEncoder:
    def __init__(
        self,
        columns_to_encode: Optional[List[str]] = None,
        for_transformer: bool = False,
        shared_embed: bool = False,
    ):
        self.columns_to_encode = columns_to_encode

        self.shared_embed = shared_embed
        self.for_transformer = for_transformer

        self.reset_embed_idx = not self.for_transformer or self.shared_embed

    def fit(self, df: pd.DataFrame) -> "LabelEncoder":
        """Creates encoding attributes"""

        df_inp = df.copy()

        if self.columns_to_encode is None:
            self.columns_to_encode = list(
                df_inp.select_dtypes(include=["object"]).columns
            )
        else:
            # sanity check to make sure all categorical columns are in an adequate
            # format
            for col in self.columns_to_encode:
                df_inp[col] = df_inp[col].astype("O")

        unique_column_vals = dict()
        for c in self.columns_to_encode:
            unique_column_vals[c] = df_inp[c].unique()

        self.encoding_dict = dict()
        if "cls_token" in unique_column_vals and self.shared_embed:
            self.encoding_dict["cls_token"] = {"[CLS]": 0}
            del unique_column_vals["cls_token"]
        # leave 0 for padding/"unseen" categories
        idx = 1
        for k, v in unique_column_vals.items():
            self.encoding_dict[k] = {
                o: i + idx for i, o in enumerate(unique_column_vals[k])
            }
            idx = 1 if self.reset_embed_idx else idx + len(unique_column_vals[k])

        self.inverse_encoding_dict = dict()
        for c in self.encoding_dict:
            self.inverse_encoding_dict[c] = {
                v: k for k, v in self.encoding_dict[c].items()
            }
            self.inverse_encoding_dict[c][0] = "unseen"

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label Encoded the categories in ``columns_to_encode``"""
        try:
            self.encoding_dict
        except AttributeError:
            raise NotFittedError(
                "This LabelEncoder instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this LabelEncoder."
            )

        df_inp = df.copy()
        # sanity check to make sure all categorical columns are in an adequate
        # format
        for col in self.columns_to_encode:  # type: ignore
            df_inp[col] = df_inp[col].astype("O")

        for k, v in self.encoding_dict.items():
            df_inp[k] = df_inp[k].apply(lambda x: v[x] if x in v.keys() else 0)

        return df_inp

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for k, v in self.inverse_encoding_dict.items():
            df[k] = df[k].apply(lambda x: v[x])
        return df
