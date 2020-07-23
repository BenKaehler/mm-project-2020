import pandas
from numpy import zeros


def fit(X: pandas.DataFrame, y: pandas.Series, **kwargs):
    """Fit a compartmental epidemiological model to COVID-19 data obtained from
    https://github.com/open-covid-19/data.

    Fits an ordinary differential equation to the y input. The independent
    variable is given by the "date" column of X, but there is other data in
    X that may be used to inform your model.

    Parameters
    ----------
    X : pandas.DataFrame
        A DataFrame containing at least the "date" columns

    y : pandas.Series
        A Series containing the response variable ("total_confirmed" cases)

    Returns
    -------
    dict
        Fitted model parameters.
    """

    # Model fitting code goes here

    params = dict(beta=2.0, gamma=0.25)  # dummy outputs

    return params


def predict(X: pandas.DataFrame, **kwargs):
    """Predict the number of confirmed cases of COVID-19 at the dates in the
    "date" column of data.

    Parameters
    ----------
    X : pandas.DataFrame
        A DataFrame containing at least a "date" column, but also containing
        covariate data that were used to fit the model. None of the
        following columns are used for prediction: "new_confirmed",
        "new_deceased", "new_recovered", "total_confirmed", "total_deceased",
        "total_recovered", "new_hospitalized", "total_hospitalized",
        "current_hospitalized", "new_intensive_care", "total_intensive_care",
        "current_intensive_care", "new_ventilator", "total_ventilator", and
        "current_ventilator".

    Keyword Arguments
    -----------------
    The model parameters.

    Returns
    -------
    array
        Array of floats giving the predicted number of confirmed COVID-19
        cases. Length is the number of rows in X.
    """

    # ODE integration code goes here
    y_predicted = zeros(len(X.index))

    return y_predicted
