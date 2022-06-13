import functools
from torch.distributions import Categorical
from torch.nn.functional import normalize
import torch
import numpy as np
import matplotlib.pyplot as plt


def constrain(p: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, min=0, max=None)
    p = normalize(p, p=1, dim=1)
    return p

def H(p: torch.Tensor):
    return Categorical(probs=p).entropy()
 

def jsd(prob_distributions, weights=None):

    if weights is None:
        n, m = prob_distributions.size()
        weights = torch.ones(n,1)/n

    # entropy of mixture
    weighted_probs = weights * prob_distributions
    mixture = weighted_probs.sum(axis=0)
    entropy_of_mixture = H(mixture)

    # sum of entropies
    weighted_entropies = weights.flatten() * H(prob_distributions)
    sum_of_entropies = weighted_entropies.sum()

    divergence = entropy_of_mixture - sum_of_entropies
    return(divergence)


def objective_function(a = 0.01, params: torch.Tensor = torch.eye(4)):
    primitives = constrain(params)
    return -(jsd(primitives) + a * H(primitives).sum())



def learn_primitives(a = 0, k = 4) -> torch.Tensor:

    obj_func = functools.partial(objective_function, a)

    params = torch.nn.parameter.Parameter(torch.rand(k, 4, requires_grad=True))
    optimizer = torch.optim.Adam([params], lr=0.001)
    
    for _ in range(1000):
        optimizer.zero_grad()
        loss = obj_func(params)
        loss.backward()
        optimizer.step()

    return constrain(params)


def radar_plot(primitives: torch.Tensor, title_str: str = ""):
    primitives = primitives.detach().numpy()

    categories = ['Right', 'Up', 'Left', 'Down']
    categories = [*categories, categories[0]]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=(primitives.shape[1]+1))

    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)

    for i, primitive in enumerate(primitives):
        primitive = [*primitive, primitive[0]]
        ax = plt.plot(label_loc, primitive, label=f'Primitive {i+1}')

    plt.title(f'Primitive policies {title_str}', size=15)

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    
    plt.legend()
    plt.show()


def main():
    for k in range(1,5):
        primitives = learn_primitives(a=0, k=k)
        radar_plot(primitives, title_str=f"K = {k}")


if __name__ == "__main__":
    main()