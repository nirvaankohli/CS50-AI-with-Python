from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

# Define the distributions for each node
rain = Categorical([[0.7, 0.2, 0.1]])  # 'none', 'light', 'heavy'

maintenance = ConditionalCategorical([
    [[0.4, 0.6],  # P(maintenance | rain='none')
     [0.2, 0.8],  # P(maintenance | rain='light')
     [0.1, 0.9]]  # P(maintenance | rain='heavy')
])

train = ConditionalCategorical([
    [[0.8, 0.2],  # P(train | rain='none', maintenance='yes')
     [0.9, 0.1]], # P(train | rain='none', maintenance='no')
    [[0.6, 0.4],  # P(train | rain='light', maintenance='yes')
     [0.7, 0.3]], # P(train | rain='light', maintenance='no')
    [[0.4, 0.6],  # P(train | rain='heavy', maintenance='yes')
     [0.5, 0.5]]  # P(train | rain='heavy', maintenance='no')
])

appointment = ConditionalCategorical([
    [[0.9, 0.1],  # P(appointment | train='on time')
     [0.6, 0.4]]  # P(appointment | train='delayed')
])

# Create a Bayesian Network and add distributions
model = BayesianNetwork()
model.add_distributions([rain, maintenance, train, appointment])  # Pass as a list

# Add edges connecting nodes
model.add_edge(rain, maintenance)
model.add_edge(rain, train)
model.add_edge(maintenance, train)
model.add_edge(train, appointment)