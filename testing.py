



import torch
import pandas as pd
import pickle
from src.libs.Datasets import LSAC_dataset
from src.preprocessing import TabPreprocessor
from src.models import TabTransformer, TabModel
from src.metrics import Accuracy, Recall, F1Score
from src.training import Trainer_fair  # Fair Training Class

with open("tab_preproc_Tab_Fair_model_LAW2.pkl", "rb") as tp:
    tab_preprocessor_new = pickle.load(tp)


# The dataset, continuous columns, categorical columns, target
df_test, cont_cols, cat_cols, target_name = LSAC_dataset('data_prepration/data_sets/test_Law_30.csv')
X_tab_test = tab_preprocessor_new.fit_transform(df_test)  # Preprocessor on test dataset
target_test = df_test[target_name].values  # Target
metrics = [Accuracy, Recall, F1Score]  # Performance Metrics

# Model
tab_transformer = TabTransformer(column_idx=tab_preprocessor_new.column_idx,
                                embed_input= tab_preprocessor_new.embeddings_input,
                                continuous_cols=tab_preprocessor_new.continuous_cols,
                                cont_norm_layer="batchnorm",
                                input_dim = 96,  
                                attn_dropout=0.1, 
                                ff_dropout = 0.0, 
                                n_blocks=1, 
                                n_heads=2, 
                                )

model_tesing = TabModel(deeptabular=tab_transformer)
trainer = Trainer_fair(model_tesing, objective="binary", metrics=metrics, tab_preprocessor=tab_preprocessor_new)
preds = trainer.predict(X_tab=X_tab_test)
print('Accuracy Score:', accuracy_score(target_test, preds))
print('F1 Score:', f1_score(target_test, preds))





