


import torch
import pandas as pd
import pickle
from src.libs.Datasets import LSAC_dataset
from src.preprocessing import TabPreprocessor
from src.models import TabTransformer, TabModel
from src.metrics import Accuracy, Recall, F1Score
from src.training import Trainer  # Training Class


df_train, cont_cols, cat_cols, target_name = LSAC_dataset('src/train_test_datasets/train_Law_70.csv') # The dataset, continuous columns, categorical columns, target

tab_preprocessor = TabPreprocessor(   # Define the Preprocessor
    embed_cols=cat_cols,
    continuous_cols=cont_cols,
    for_transformer=True,
    with_cls_token=True,
)

# Performance Metrics
metrics = [Accuracy, Recall, F1Score]

# Define the Preprocessor
tab_preprocessor = TabPreprocessor(
    embed_cols=cat_cols,
    continuous_cols=cont_cols,
    for_transformer=True,
    with_cls_token=True,
)


# preprocessinf the train set
X_tab = tab_preprocessor.fit_transform(df_train)

# Target
target = df_train[target_name].values


# Tab Transformer Model
tab_transformer = TabTransformer(column_idx=tab_preprocessor.column_idx,
                                embed_input= tab_preprocessor.embeddings_input,
                                continuous_cols=tab_preprocessor.continuous_cols,
                                cont_norm_layer="batchnorm",
                                input_dim = 96,  
                                attn_dropout=0.1, 
                                ff_dropout = 0.0, 
                                n_blocks=1, 
                                n_heads=2, 
                                )
model = TabModel(deeptabular=tab_transformer)
trainer = Trainer(model, objective="binary", metrics=metrics)
trainer.fit(X_tab=X_tab, target=target, n_epochs=10, batch_size=256, val_split=None)



# Save the trained model
torch.save(model, "Saved_Models/Tab_model_saved.pt")
torch.save(model.state_dict(), "Saved_Models/Tab_model_state_dict_saved.pt")
# Save the preprocessor
with open("tab_preproc_Tab_model_LAW2.pkl", "wb") as dp:
    pickle.dump(tab_preprocessor, dp)







