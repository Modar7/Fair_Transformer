import pandas as pd


def LSAC_dataset(path):
    num_col_names = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', ]
    cat_col_names = ['fulltime','fam_inc' ,'male', 'race', 'tier']
    target_name = ["pass_bar"]

    df_data = pd.read_csv(path)

    return df_data, num_col_names, cat_col_names, target_name

