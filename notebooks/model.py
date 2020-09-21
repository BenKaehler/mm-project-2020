from datetime import date

import pandas
from numpy import zeros, inf
from scipy.integrate import odeint
from scipy.optimize import curve_fit

def dydt(y, t, k, C, X):
    # Want X in here
    return k/C*y*(C - y)

def extract_t_from(X):
    dates = X['date'].apply(date.fromisoformat).tolist()
    return [(d - dates[0]).days for d in dates]

def get_model(X):
    def model(t, y0, k, C):
        return odeint(dydt, y0, t, args=(k, C, X))[:,0]
    return model

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
    t = extract_t_from(X)
    model = get_model(X)
    (y0, k, C), _ = curve_fit(model, t, y, bounds=((0, 0, 0), (inf, inf, inf)))
    return dict(y0=y0, k=k, C=C)


def predict(X: pandas.DataFrame, y0=1, k=0.0001, C=6000, **kwargs):
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

    t = extract_t_from(X)
    model = get_model(X)
    y_predicted = model(t, y0, k, C)
    
    return y_predicted