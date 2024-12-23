#%%
import numpy as np
import scipy.linalg
from scipy.stats import binom, nbinom
import matplotlib.pyplot as plt
import math

def compute_P(max_patients,p,q,r):
    # Transition matrix for the Markov chain
    P = np.zeros((max_patients+1, max_patients+1))

    # Fill the transition matrix
    for i in range(max_patients+1):
        for j in range(max_patients+1):
            prob = 0
            for stay in range(i+1): # Staying patients 
                for new in range(max_patients+1): # new patients
                    if (stay + new) == j: # When it mathces the correct column
                        prob += binom.pmf(stay, i, 1-p) * nbinom.pmf(new, r, q)

            P[i, j] = prob

    P[:,10] += 1-P.sum(axis=1)

    return P

from matplotlib.colors import LinearSegmentedColormap

def plot_P(P):
    # Create a custom colormap from white to red
    cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Set the figure size
    plt.imshow(P, aspect="auto", cmap=cmap)  # Plot the heatmap with the custom colormap

    # Add color bar
    plt.colorbar(label="Transition Probabilities")

    # Add labels, title, and ticks
    plt.xlabel("State j")
    plt.ylabel("State i")
    plt.title("Transition Probability Matrix")

    # Add ticks for each state
    num_states = len(P)
    plt.xticks(ticks=range(num_states), labels=range(num_states))
    plt.yticks(ticks=range(num_states), labels=range(num_states))

    # Optionally annotate the cells with their values
    for i in range(num_states):
        for j in range(num_states):
            if P[i][j] > 0:  # Only annotate non-zero values
                plt.text(j, i, f"{P[i][j]:.2f}", ha="center", va="center", color="black")

    # Show the plot
    plt.show()


# Parameters
p = 1/10
q = 2/5
r = 2
max_patients = 10

#%% Question 2

P = compute_P(max_patients,p,q,r)

# Initial state (no patients initially) 
initial_state = np.zeros(max_patients+1)
initial_state[0] = 1

# Calculate the state distribution after 12 months
state_distribution = initial_state @ np.linalg.matrix_power(P, 12)
print(sum(state_distribution))

# Calculate the mean number of patients
mean_patients = sum(i * state_distribution[i] for i in range(max_patients+1))
print(mean_patients)

#%% Question 3 Limiting distribution 

# Finding the limit distribution
# Solve (P^T - I) @ pi = 0 with the constraint sum(pi) = 1
I = np.eye(max_patients + 1)
A = P.T - I
A[-1] = 1  # Replace the last row to enforce the constraint sum(pi) = 1

b = np.zeros(max_patients + 1)
b[-1] = 1  # Right-hand side of the equation for the constraint

# Solve the linear system to find the limit distribution
pi_limit = np.linalg.solve(A, b)

# Print the limit distribution
print("limit distribution:", pi_limit)

# Verify that pi satisfies pi @ P = pi
print("Check: pi @ P - pi =", np.allclose(pi_limit @ P, pi_limit))

# Calculate the mean and variance
mean_limit = sum(n * pi_limit[n] for n in range(max_patients + 1))

variance_limit = sum((n - mean_limit)**2 * pi_limit[n] for n in range(max_patients + 1))

# Print the results
print("limit distribution:", pi_limit)
print("Mean number of patients:", mean_limit)
print("Variance of the number of patients:", variance_limit)

# Verify that pi satisfies pi @ P = pi
print("Check: pi @ P - pi =", np.allclose(pi_limit @ P, pi_limit))


#%% Question 4 More than 2 years to absorption

A = P[1:-1,1:-1].copy()
I = np.eye(len(A))
b = np.ones(A.shape[0])

v_i = np.linalg.solve(I-A, b)

print("time to absorption starting at state 4: ",v_i[4])

# Inserting the absorption states
P[0] = 0
P[10] = 0
P[0,0] = 1
P[10,10] = 1

P24 = np.linalg.matrix_power(P, 24)

# first method to find probability of absorption after 24 months
initial_state = np.zeros(P.shape[0])
initial_state[4] = 1 
P_state24 = initial_state @ P24
print(1-(P_state24[0]+P_state24[-1]))

#%% Quesiton 4 method using law of total probability

Q = P[1:-1,1:-1].copy()
I = np.eye(len(A))
Q24 = np.linalg.matrix_power(Q, 24)

b0 = P24[1:-1,0]
b10 = P24[1:-1,-1]

u0 = np.linalg.solve(I-Q24,b0)
u10 = np.linalg.solve(I-Q24,b10)
print(u0[4])
print(u10[4])

print(1-(u10[4]+u0[4]))

# %% Gamblers ruin Question 5

P = compute_P(max_patients,p,q,r)
P[0] = 0
P[10] = 0
P[0,0] = 1
P[10,10] = 1

P24 = np.linalg.matrix_power(P, 24)
print(P24[5,10]/(P24[5,10]+P24[5,0]))
print(P[5,10]/(P[5,10]+P[5,0]))

# Page 118 eq. 3.42
test = 1-((P[5,0]/P[5,10])**5-(P[5,0]/P[5,10])**10)/(1-(P[5,0]/P[5,10])**10)

print(test)

b10 = P[1:-1,-1]
u10 = np.linalg.solve(I-Q,b10)
print(u10[4])

# %% Question 6 

conditional_prob = u10[4]/(u10[4]+u0[4])
print(conditional_prob)


#%% ------------------------PART 2-------------------------------

# %% Question 7

def compute_A(lamb,mu,size):
    A = np.zeros([size,size])
    A[0,0] = -lamb
    A[0,1] = lamb

    for i in range(1,size):
        for j in range(1,size):
            lamb = 1 / 3  
            mu = 1 / 8    
            if i == j:

                if i>3:
                    mu*=3
                else:    
                    mu*=i
                
                if i==(size-1) and j == (size-1):
                    A[i,j-1] = mu
                    A[i,j] =-mu 
                else:
                    A[i,j-1] = mu
                    A[i,j]   = -(mu+lamb)
                    A[i,j+1] = lamb 

    return A

lamb = 1 / 3  
mu = 1 / 8    
s = 3         
rho = lamb / (s * mu)

A = compute_A(lamb,mu,size=20)

plot_P(A)

import scipy
P24 = scipy.linalg.expm(A*24)

plot_P(P24)
print(P24.sum(axis=1))

print(np.sum(P24[0,7:])) 

# %% Question 8

A = compute_A(lamb,mu,size=7)
S = A[0:-1,0:-1]
alpha = np.zeros(S.shape[0])
alpha[0] = 1
ones = np.ones_like(alpha)
mean = -alpha@np.linalg.inv(S)@ones
var = 2*alpha@np.linalg.inv(S)@np.linalg.inv(S)@ones-(alpha@np.linalg.inv(S)@ones)**2
print(mean,var)

# %% Question 9 

A = compute_A(lamb,mu,size=7)

A_system = A.T[0:-1,0:-1].copy()
A_system[-1] = 1
max_patients = A_system.shape[0]

b = np.zeros(max_patients)
b[-1] = 1
pi_limit = np.linalg.solve(A_system, b)

# Verify that pi satisfies pi @ P = pi
print("Check: ", np.allclose(pi_limit @ A_system, b,rtol=1,atol=1))

# Calculate the mean and variance
mean_limit = sum(n * pi_limit[n] for n in range(max_patients))
print(mean_limit)

# %% Question 10 Se udledninger i hånden

prob_exceed_24 = np.exp(-24*(mu*3-lamb))
print(prob_exceed_24)

# %% Question 11 Mean number of patients in trigging

lamb = 9/60
normal_time = 5
shape = 4/3
mean_gamma = 10
lamb_gamma = shape/mean_gamma
p=1/10
tau_2 = p**2*shape/(lamb_gamma**2)
nu = 6
rho = lamb*nu

L = rho+(lamb**2*tau_2+rho**2)/(2*(1-rho))
print(L)

#%% Question 12 NOT DONE


#%% --------------------------- PART 3 ------------------------------

#%% Question 13 

# Parameters
n = 6  
p = 0.5  

# Re-admissions in the second half of the year
prob_at_least_4 = 1-(binom.pmf(3, n, p) + binom.pmf(2, n, p) + binom.pmf(1, n, p)+ binom.pmf(0, n, p))
print(prob_at_least_4)

#%% Question 14

# Given values
T = 1/2  # mean duration of township care period in months
S = 3    # mean duration of stable period in months
U = 1/4 # mean duration of unstable period in months
p_s = 1/4  # probability of entering stable period
p_u = 3/4  # probability of entering unstable period

lambda_s = 1/S
lambda_u = 1/U

# Weighted sum (time in the system)
process1_time = T + p_s * S + p_u * U

# Number of times in the system during 2 years
number_of_time_2_years = 24/process1_time

# Calculating expected re-admissions per time
E_re_admissions_per_cycle = p_s * lambda_s * process1_time + p_u * lambda_u * process1_time

total_re_admissions = number_of_time_2_years * E_re_admissions_per_cycle

print(total_re_admissions)

# %% Question 15 

number_of_time_3_years = 36/process1_time
lamb = number_of_time_3_years*E_re_admissions_per_cycle
P_morethan24 = 1-sum(lamb**k*np.exp(-lamb)/math.factorial(k) for k in range(25))
print(P_morethan24)

#%% Question 16
t = 2
lamb = 1/3
# Second answer This is the correct answer 
P_first_2months = 1-np.exp(-2/3)
print(P_first_2months)

# %% Question 17 

# Done in Latex

#%% Question 18

import numpy as np

# Define the transition matrix
P = np.array([
    [0, 3/4 * 3/4, 3/4 * 1/4, 1/4, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 2/3, 0, 1/3, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 3/4, 1/4],
    [1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0]
])

I = np.eye(P.shape[0])
A = P.T - I
A[-1, :] = 1  # Replace the last row to enforce the constraint sum(pi) = 1
b = np.zeros(P.shape[0])
b[-1] = 1  # Right-hand side of the equation for the constraint

plot_P(A)
# Solve the linear system to find the limit distribution
pi_limit = np.linalg.solve(A, b)

print("Steady-state distribution (pi):", pi_limit)

# New method

time_hos = 1/40+3/4*(3/4* 1/3+1/4*(1/6+2/3*1/3))
print(time_hos)

time_gen = 3/4*(3/4* 1/3+1/4* 2/3*1/3)
print(time_gen)

time_total = process1_time+time_hos

frac_gen = time_gen/time_total
print("frac gen",frac_gen)

frac_hos = time_hos/time_total
print("frac hos",frac_hos)

#%% --------------------- PART 4---------------------------------------------

#%% Question 19 

# Done in Latex

#%% Question 20 

# Done in Latex

#%% Question 21

# Done in Latex

#%% Quesiton 22

# Done in Latex

#%% Question 24

import numpy as np

mu = -2
sigma2 = 4
x = 6
a = 0
b = 10

ub = (np.exp(-2*mu*x/sigma2)-np.exp(-2*mu*a/sigma2))/(np.exp(-2*mu*b/sigma2)-np.exp(-2*mu*a/sigma2))
ua = 1-ub

exptedb = (1/mu)*(ub*(b-a)-(x-a))
expteda = (1/mu)*(ua*(a-b)-(x-b))
print(exptedb)
print(expteda)

total_expected = ub*exptedb+ua*expteda
print(total_expected)



