import matplotlib


def set_matplotlib_config():
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams["legend.loc"] = 'upper right'
    matplotlib.rcParams['axes.titlesize'] = 'x-large'
    matplotlib.rcParams['axes.labelsize'] = 'x-large'
    matplotlib.rcParams['legend.fontsize'] = 'x-large'
    matplotlib.rcParams['xtick.labelsize'] = 'x-large'
    matplotlib.rcParams['ytick.labelsize'] = 'x-large'
    params= {'text.latex.preamble' : [r'\usepackage{amsmath}', 
        r'\usepackage{amssymb}']}
    matplotlib.rcParams.update(params)
