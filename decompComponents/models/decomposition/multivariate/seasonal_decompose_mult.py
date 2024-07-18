from statsmodels.tsa.seasonal import seasonal_decompose
from decompComponents.functions import get_feature_with_dates


def decompose_multivariate_seasonal(selected_node_data, model="additive"):

    trend_data = selected_node_data.copy()
    seasonal_data = selected_node_data.copy()
    residue_data = selected_node_data.copy()

    # Preprocessing the data
    for i in range(len(selected_node_data.columns)):
        selected_node_feature_data = get_feature_with_dates(
            df=selected_node_data, feature_index=i)

        # stl = STL(selected_node_feature_data,robust=True,)
        # res = stl.fit()
        res = seasonal_decompose(selected_node_feature_data, model=model)

        seasonal, trend, residue = res.seasonal, res.trend, res.resid
        trend_data.iloc[:, i] = trend
        seasonal_data.iloc[:, i] = seasonal
        residue_data.iloc[:, i] = residue

    # remove first 3 and last 3 values as they are NAN
    trend_data = trend_data[3:-3]
    seasonal_data = seasonal_data[3:-3]
    residue_data = residue_data[3:-3]

    return trend_data, seasonal_data, residue_data
