import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import LabelEncoder


class feature_selection:

    def corr(data, self=None):
        data = self.Feature_Encoder(data)
        col_names = data.columns
        param = []
        correlation = []
        abs_correlation = []
        for c in col_names:
            if c != 'income':
                if len(data[c].unique()) <= 2:
                    corr = spearmanr(data['income'], data[c])[0]
                else:
                    corr = pearsonr(data['income'], data[c])[0]
                param.append(c)
                correlation.append(corr)
                abs_correlation.append(abs(corr))
        param_df = pd.DataFrame({'correlation': correlation, 'parameter': param, 'abs_correlation': abs_correlation})
        param_df = param_df.sort_values(by=['abs_correlation'], ascending=False)
        # get features with correlation > 0.1
        param_df = param_df[param_df['abs_correlation'] < 0.1]
        # put features names in list
        param_df = param_df['parameter'].tolist()
        return param_df



    def Feature_Encoder(X,self=None):
        cols = X.columns
        for c in cols:
            if X[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(X[c].values))
                X[c] = lbl.transform(list(X[c].values))
        return X