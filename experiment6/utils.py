import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize

def gaussian(p,x):
    A, mu, sigma = p
    return A / (sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu) ** 2 / ( 2 * sigma **2))

def bigaussian(p, x):
    return gaussian(p[:3], x) + gaussian(p[3:], x)

def bigaussianlinear(p, x):
    return linear(p[:2],x) + gaussian(p[2:5], x) + gaussian(p[5:8], x)

def linear(p,x):
    return p[0]*x + p[1]

def gaussianlinear(p,x):
    return linear(p[:2], x) + gaussian(p[2:5], x)

def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err

def exponential(p, x):
    return p[0] * np.exp(-p[1] * x)

def data_fit(p0, func, xvar, yvar, err, tmi=0):
    
    # The code below defines our data fitting function.
    # Inputs are:
    # initial guess for parameters p0
    # the function we're fitting to
    # the x,y, and dy variables
    # tmi can be set to 1 or 2 if more intermediate data is needed

    try:
        fit = optimize.least_squares(residual, p0, args=(func,xvar, yvar, err), verbose=tmi)
    except Exception as error:
        print("Something has gone wrong:", error)
        return p0, np.zeros_like(p0), np.nan, np.nan
    pf = fit['x']

    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))          
        # This computes a covariance matrix by finding the inverse of the Jacobian times its transpose
        # We need this to find the uncertainty in our fit parameters
    except:
        # If the fit failed, print the reason
        print('Fit did not converge: ', fit['status'])
        print(fit['message'])
        return pf, np.zeros_like(pf), np.nan, np.nan
        #You'll be able to plot with this, but it will not be a good fit.

    chisq = sum(residual(pf, func, xvar, yvar, err) **2)
    dof = len(xvar) - len(pf)
    red_chisq = chisq/dof
    pferr = np.sqrt(np.diagonal(cov)) # finds the uncertainty in fit parameters
    
    #print('Converged.')
    
    return pf, pferr, chisq,dof

def fit(X, Y, Y_err, fit_range=None, ax=None, ):

    # Fit Data
    if fit_range is not None: fit_lower, fit_upper = fit_range
    
    x = X[fit_lower:fit_upper] if fit_range is not None else X
    y = Y[fit_lower:fit_upper] if fit_range is not None else Y
    y_err = Y_err[fit_lower:fit_upper] if fit_range is not None else Y_err
    
    print(f'Fit Region: ({x.min()}, {x.max()})')

    params_0 = [0, y.mean()]

    params, params_err, chisq, dof = data_fit(params_0, linear, x, y, y_err)
    
    # Plot Data

    if ax is None: fig, ax = plt.subplots(figsize=(5, 5))

    ax.errorbar(
        X, Y, Y_err, 
        color='red', 
        fmt='.', 
        label='Data'
    )
    
    # Plot Fit Line

    x_padding = (x.max() - x.min()) * 0.1
    linspace = np.linspace(x.min()-x_padding, x.max()+x_padding)
    ax.plot(linspace, linear(params, linspace), color='black', linestyle='dashed', label='Fitted Curve')
    
    # Shade Fit Region
    
    if fit_range is not None: ax.axvspan(x[-1], x[0], alpha=0.25, label='Fit Region')
    
    # Set Labels

    ax.grid(alpha=0.5)
    ax.legend(loc='upper left')
    
    return params, params_err, chisq, dof

def parse_usx_csv(fname):
    
    with open(fname) as f:
        lines = f.readlines()


    for i, line in enumerate(lines):
        if line.startswith('Channel,Energy,Counts'): break
        if 'Real Time' in line: 
            time = float(line.split(',')[-1])

    data = np.array([line.split(',') for line in lines[i+1:]])

    return time, data[:, 0].astype(int), data[:, 2].astype(int)
