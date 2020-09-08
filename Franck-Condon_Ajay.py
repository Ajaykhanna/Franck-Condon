#!/usr/bin/env python
# coding: utf-8

# ## Computing Franck Condon Spectra for CO Molecule
# ## Written by: Khanna Ajay || Date Completed: Feb.10.2020 || Lab: Dr. C.M. Isborn

# In[40]:


#importing libraries
import numpy as np
from numpy.polynomial.hermite import hermval
from numpy import math
import pandas as pd
from pandas import DataFrame as df

# Increases the Width of Current Jupyter Notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
#pd.set_option('display.max_colwidth', -1)  # or 199

# Import Matplot Library
import matplotlib.pyplot as plt
import matplotlib as matplot
from matplotlib.pyplot import figure
   
import chart_studio.plotly as plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from plotly.subplots import make_subplots
import plotly
#plotly.tools.set_credentials_file(username='Samdig', api_key='0k2OH9PK2zjVYpASiiFY')

# Offline Mode
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Interaction Mode

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# ## Equation Involved
# $$\Psi_\nu(x) = N_\nu H_\nu(\alpha x) e^{(-\frac{\alpha^2 x^2}{2})} $$
# 
# $$\alpha = (\frac{\mu k_f}{\hbar^2})^\frac{1}{4} $$
# $$Also: \alpha = (\mu^2\sqrt{\omega})^\frac{1}{4} $$
# $$\omega = \sqrt{\frac{k_f}{\mu}}$$
# 
# $$N_\nu = (\frac{\alpha}{2.^\nu \nu!\pi^\frac{1}{2}})^\frac{1}{2}$$
# 
# $$\nu = 0, 1, 2, 3, .... \infty $$
# 
# $$H_\nu(αx) = (-1)^{\nu} \exp(\alpha^2x^2) \frac{d^\nu}{d \mathbf{x^\nu}} exp(-\alpha^2x^2) $$

# In[41]:


# System = Carbon monoxide (CO)
# Predefined Variables, All Units are in Atomic Units

h_bar = 1.
v = np.arange(0,50)                                                                                        # Range of Vibrational Quantum Number

atom_1_mass = 12.000000000                                                                                 # Mass of Atom-1
atom_2_mass = 15.994914640                                                                                 # Mass of Atom-2
m_reduced = (atom_1_mass*atom_2_mass/(atom_1_mass + atom_2_mass))                                          # Reduced Mass
 

# System's Parameter
force_constant_CO = 1857                                                                                   # Units: N/meter

k_gs = (force_constant_CO * 2.293710449e+17 )/((1.889725989e+10)**2)                                       # Force Constant of System in Ground State, Units: Hartree/Bohr^2
k_ex = k_gs - ((5./100) * k_gs)                                                                           # Force Constant of System in Excite State

α_gs = np.power((k_gs*m_reduced)/h_bar, (1./2.))                                                           # Unitless Parameter
α_ex = np.power((k_ex*m_reduced)/(h_bar**2), (1./2.))

ω_gs = np.sqrt(k_gs/m_reduced)                                                                             # Angular Frequencies of States
ω_ex = np.sqrt(k_ex/m_reduced)

# Electronic Energy for Ground and Excited State 
E_gs = -113.317323                                               # Grounds State Energy - Calculated Using DFT/B3LYP 6-31+g(d,p), This is With Zero Point Correction (Units: Hartree)
E_ex = -113.025632                                               # Excited State Energy - Calculated Using TDDFT/B3LYP 6-31+g(d,p), root=1, This is With Zero Point Correction (Units: Hartree)
# Difference in E = -113.025632 + 113.317323 = 0.291691 Hartree --> eV = 27.2114 * 0.291691 = 7.93732
# Differnce in R = R_ex - R_gs = 1.24202 - 1.13722 = 0.1048 Angs = 0.19804 Bohr
#E_ex = -113.296930                                                 

#T = 298.15                                                                                # Temperature (in Kelvin)
k_b = 1.380649 * (10**(-23))                                                                 # In JK^-1

# Conversion Factors
joule_2_hatree = 2.293710449e+17
#print(ω_gs, ω_ex)
#ω_ex = 100.


# In[42]:


# Calculating Vibrational Energy of the state
def vibrational_energy(v,h_bar,ω):
    E_v = (v + (1./2.)) * h_bar * ω                                                                        # Energy in Atomic Units--> Hartree
    return E_v

# Vibrartional Energies of Ground & Excited State
E_v_gs = vibrational_energy(v,h_bar,ω_gs)
E_v_ex = vibrational_energy(v,h_bar,ω_ex)


# Total Energy of Ground and Excited State
E_gs_total = E_v_gs + E_gs
E_ex_total = E_v_ex + E_ex

#pd.DataFrame([E_gs_total, E_ex_total]).T


# In[43]:


# Calculating Normalization Constant for the Wavefunction
def normalization(v,α):
    N_v = math.sqrt(α/(2.**v * math.factorial(v) * math.sqrt(math.pi)))
    return N_v
    
N_v_gs = []                                                                                                   # Normalization Constant Array
for i in range(len(v)):
    N_v_gs.append(normalization(v[i], α_gs))

N_v_ex = []                                                                                                   # Normalization Constant Array
for i in range(len(v)):
    N_v_ex.append(normalization(v[i], α_ex))


# In[44]:


# Calculating Normalization Constant for the Wavefunction
def normalization(v,α):
    N_v = math.sqrt(α/(2.**v * math.factorial(v) * math.sqrt(math.pi)))
    return N_v
    
N_v_gs = []                                                                                                   # Normalization Constant Array
for i in range(len(v)):
    N_v_gs.append(normalization(v[i], α_gs))

N_v_ex = []                                                                                                   # Normalization Constant Array
for i in range(len(v)):
    N_v_ex.append(normalization(v[i], α_ex))


# In[45]:


# Calculating 'Physicsts' Hermite Polynomial
d = 0.
x = np.arange(-40.,40.,.01)
Hermite_gs = []                                                                                               # Hermite Polynomical Array
coef = np.zeros(v.shape)                                                                                    # Coeffcients of Hermite Polynomial
coef[0] = 1.                                                                                               # Changing First Element of Coffecient to 1

# For Loop to Calculate Hermite Polynomials for Every Vibrational Quantum Number
for i in range(49):
    Hermite_gs.append(hermval(x=α_gs*(d-x),c=coef))                                                              # Filling Hermite Array with Hermite Polynomical at x=(α*(x-d))
    coef[i] = 0                                                                                            # Replacing Coeffcient 'ith' place with zero
    coef[i+1] = 1                                                                                          # Replace Coeffcient 'i+1th' place with 1
Hermite_gs.append(hermval(x=α_gs*(d-x),c=coef))                                                                  # Calling Hermite for the 50th vibrational number

gauss_func = []
for i in x:
    g = np.exp(-((α_gs*(d-i))**2)/2)
    gauss_func.append(g)
    
ψ_gs = []
for i in v:
    h = gauss_func * Hermite_gs[i] *  N_v_gs[i]
    ψ_gs.append(h)


# In[46]:


# Displacement and Hermite Polynomials
d = 0.5                                                                                                    # Shift in 'X-Value'
Hermite_es = []                                                                                            # Hermite Polynomical Array
coef = np.zeros(len(v))                                                                                    # Coeffcients of Hermite Polynomial
coef[0] = 1.                                                                                               # Changing First Element of Coffecient to 1

# For Loop to Calculate Hermite Polynomials for Every Vibrational Quantum Number
for i in range(49):
    Hermite_es.append(hermval(x=α_ex*(d-x),c=coef))                                                              # Filling Hermite Array with Hermite Polynomical at x=(α*(x-d))
    coef[i] = 0                                                                                            # Replacing Coeffcient 'ith' place with zero
    coef[i+1] = 1                                                                                          # Replace Coeffcient 'i+1th' place with 1
Hermite_es.append(hermval(x=α_ex*(d-x),c=coef))                                                                  # Calling Hermite for the 50th vibrational number

gauss_func_es = []
for i in x:
    g_ = np.exp(-((α_ex*(d-i))**2)/2)
    gauss_func_es.append(g_)
    
ψ_es = []
for i in v:
    h_ = gauss_func_es * Hermite_es[i] *  N_v_ex[i]
    ψ_es.append(h_)
    
    
# Difference
diff_E = E_ex_total - E_gs_total[0]
overlap = np.zeros(len(v))
for i in range(len(v)):
    overlap[i] = (np.trapz((ψ_gs[0]*ψ_es[i]),x,dx=0.01))**2

pd.DataFrame([overlap,diff_E])


# In[47]:


# Harmonic Potential Energy Diagrams for Diatomic Molecules
z = np.linspace(-200,200,50)
y_gs = E_gs_total + (ω_gs**2 * m_reduced)* (z)**2 #(1./2.) * np.sqrt(ω_gs**2 * m_reduced)* z**2
y_ex = E_ex_total + (ω_ex**2 * m_reduced)* (z-d)**2 # (1./2.) * np.sqrt(ω_ex**2 * m_reduced)* z**2
plt.figure(figsize=(18,7))
plt.xlabel('Displacement')
plt.ylabel('Potential Energy')
plt.ylim(top=12000)
#markerline, stemlines, baseline = plt.stem(z, y, '-.')
#plt.setp(baseline, 'color', 'r', 'linewidth', 2)
plt.plot(z,y_gs,'-o')
plt.plot(z,y_ex, '-*')
plt.legend(['Ground Ocsillator', 'Excited Oscillator'])
plt.show()


# In[48]:


# Setting Up Trace
trace1 = go.Scatter(
    x=x,
    y=ψ_gs[0],
    mode='lines',
    name='Ground Wavefunction',
    marker=dict(
        #color='rgb(220, 20, 60)'
    )
)

trace2 = go.Scatter(
    x=x,
    y=ψ_es[0],
    mode='lines',
    name='Excited Wavefunction',
    marker=dict(
        #color='rgb(100, 149, 237)'
    )
)
#Setting up the Layout for the Graphs

layout = go.Layout(
    template='plotly_white',
    title='Wavefunction Overlap of CO Molecule',
    xaxis=dict(
        title='Range of X',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Vibrational Wavefunction',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='jupyter-basic_bar')


# FC Plot

plt.figure(figsize=(28,10))
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.title('Franck-Condon Spectra at Zero Temperature')
markerline, stemlines, baseline = plt.stem(diff_E, overlap, '-.', use_line_collection=True)
plt.setp(baseline, 'color', 'r', 'linewidth', 2)
plt.show()


# In[49]:


# Boltzman Distribution
T = 300*298.15
print("Current Temperature(K): ", T)
boltzman = np.zeros(v.shape)
for i in range(len(v)):
    boltzman[i] = np.exp(-(i*h_bar*ω_gs)/(T*k_b*joule_2_hatree))
pd.DataFrame(boltzman).T


# In[50]:


# Testing Boltzman Contribution
diff_E = np.zeros([len(v),len(v)])
for i in range(len(v)):
    for j in range(len(v)):
        diff_E[i][j] = E_ex_total[j] - E_gs_total[i]
overlap = np.zeros([len(v),len(v)])
for i in range(len(v)):
    for j in range(len(v)):
        overlap[i][j] = boltzman[i]*(np.trapz((ψ_gs[i]*ψ_es[j]),x,dx=0.01))**2


# In[51]:


# Plotting BZman Distribution
x = (diff_E[i] for i in range(len(v)))
y = (overlap[j] for j in range(len(v)))

#lines = [y1,y2,y3]
#colors  = ['r','g','b']
#labels  = ['RED','GREEN','BLUE']

# fig1 = plt.figure()
plt.figure(figsize=(28,10))
#plt.title('Franck Condon at Finite Temperature @: T')
plt.title('Finit Temperature Franck-Condon Spectra of CO @T= {}K'.format(T))
for x,y in zip(x,y):  
    #plt.plot(x,y)
    #plt.legend(labels)
    plt.xlabel('Energy')
    plt.ylabel('Intensity')
    markerline, stemlines, baseline = plt.stem(x, y, '--o', use_line_collection=True)
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
plt.show()


# In[19]:


# Lollipop Chat
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

# Generate a random signal
np.random.seed(42)
random_signal = np.random.normal(size=100)

# Offset the line length by the marker size to avoid overlapping
marker_offset = 0.04

def offset_signal(signal, marker_offset):
    if abs(signal) <= marker_offset:
        return 0
    return signal - marker_offset if signal > 0 else signal + marker_offset

data = [
    go.Scatter(
        x=list(range(len(random_signal))),
        y=random_signal,
        mode='markers',
        marker=dict(color='red')
    )
]

# Use the 'shapes' attribute from the layout to draw the vertical lines
layout = go.Layout(
    template='plotly_dark',
    shapes=[dict(
        type='line',
        xref='x',
        yref='y',
        x0=i,
        y0=0,
        x1=i,
        y1=offset_signal(random_signal[i], marker_offset),
        line=dict(
            color='grey',
            width=1
        )
    ) for i in range(len(random_signal))],
    title='Lollipop Chart'
)

# Plot the chart
fig = go.Figure(data, layout)

pyo.iplot(fig)


# In[ ]:




