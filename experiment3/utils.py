import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

def linear(p,x):
    return p[0]*x + p[1]

def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err

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

    print()

    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))          
        # This computes a covariance matrix by finding the inverse of the Jacobian times its transpose
        # We need this to find the uncertainty in our fit parameters
    except:
        # If the fit failed, print the reason
        print('Fit did not converge')
        print('Result is likely a local minimum')
        print('Try changing initial values')
        print('Status code:', fit['status'])
        print(fit['message'])
        return pf, np.zeros_like(pf), np.nan, np.nan
        #You'll be able to plot with this, but it will not be a good fit.

    chisq = sum(residual(pf, func, xvar, yvar, err) **2)
    dof = len(xvar) - len(pf)
    red_chisq = chisq/dof
    pferr = np.sqrt(np.diagonal(cov)) # finds the uncertainty in fit parameters
    
    print('Converged with chi-squared {:.2f}'.format(chisq))
    print('Number of degrees of freedom, dof = {:.2f}'.format(dof))
    print('Reduced chi-squared {:.2f}'.format(red_chisq))
    print()
    Columns = ["Parameter #", "Initial guess values:", "Best fit values:", "Uncertainties in the best fit values:"]
    
    print(
        '{:<11}'.format(Columns[0]), '|', 
        '{:<24}'.format(Columns[1]), '|',
        '{:<24}'.format(Columns[2]), '|', 
        '{:<24}'.format(Columns[3])
    )
    
    for num in range(len(pf)):
        print(
            '{:<11}'.format(num),'|', 
            '{:<24.3e}'.format(p0[num]), '|', 
            '{:<24.3e}'.format(pf[num]), '|', 
            '{:<24.3e}'.format(pferr[num])
        )
    
    return pf, pferr, chisq,dof

def fit(X, Y, Y_err, fit_lower, fit_upper, xlabel, ylabel, title):

    # Fit Data
    
    x = X[fit_lower:fit_upper].to_numpy()
    y = Y[fit_lower:fit_upper].to_numpy()
    y_err = Y_err[fit_lower:fit_upper].to_numpy()

    params_0 = [1, 10]

    params, params_err, chisq, dof = data_fit(params_0, linear, x, y, y_err)
    
    # Plot Data

    fig, ax = plt.subplots(figsize=(5, 5))

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
    
    ax.axvspan(x[-1], x[0], alpha=0.25, label='Fit Region')
    
    # Add Text

    txt = '$\\log D = \\gamma \\cdot \\log (t_d - t) + \\log A$ \n'
    txt += f'$\\gamma = {params[0]:.2f} \\pm {params_err[0]:.2f}$ \n'
    txt += f'$\\log A = {params[1]:.2f} \\pm {params_err[1]:.2f}$ \n'
    txt += f'$\\chi^2 = {chisq:.2f}$ \n'
    txt += f'DOF$ = {dof}$'
    ax.text(0.02, 0.775, txt, transform=ax.transAxes , fontsize=8, va='top')
    
    # Set Labels

    ax.grid(alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    return params, params_err, chisq, dof