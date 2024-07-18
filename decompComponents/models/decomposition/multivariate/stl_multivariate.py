from statsmodels.tsa.seasonal import STL
from decompComponents.functions import get_feature_with_dates

def decompose_multivariate(selected_node_data) :

    trend_data = selected_node_data.copy()
    seasonal_data = selected_node_data.copy()
    residue_data = selected_node_data.copy()

    # Preprocessing the data
    for i in range(len(selected_node_data.columns)) :
        selected_node_feature_data = get_feature_with_dates(df=selected_node_data, feature_index=i)
        
        stl = STL(selected_node_feature_data,robust=True,)
        res = stl.fit()
        # res = seasonal_decompose(selected_node_feature_data, model='additive')

        seasonal,trend,residue = res.seasonal, res.trend, res.resid
        trend_data.iloc[:,i] = trend
        seasonal_data.iloc[:,i] = seasonal
        residue_data.iloc[:,i] = residue

    return trend_data, seasonal_data, residue_data