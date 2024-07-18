from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def seasonal_decompose_multiplicative(selected_node_feature_data) :

    res = seasonal_decompose(selected_node_feature_data, model='multiplicative')
    res.plot()
    plt.show()

    return res.seasonal, res.trend, res.resid