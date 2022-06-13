[[Compositional GVFs]]

# Compositional GVFs
## Learning primitive policies
In the most general case, the argument is that agents need to have a clear separation between the RL problem from other problems such as learning dynamics of movement. For example, a tennis player spends years practicing their swing in order to learn the dynamics of the movements of the swing. There is a difference between when they don't win a point because the swing was terrible or whether the strategy was bad, and the agent needs to know this distinction in order to learn. Simply throwing all the complexity in one huge optimization problem like RL does seems like the dumbest way to proceed. 

### Learning with RL
Add a penalty term to the reinforcement learning objective for the [[Compositional GVFs]] framework and then optimize it all together.

### Unsupervised approaches for learning primitives
[[CGVFs learning primitives PoA]]

#### State-independent primitives with discrete actions
##### Problem layout
Let there be $K$ primitive policies, $\rho_1, ..., \rho_K$, that are independent of the states and defined as probability distributions over the space of actions. In the case of the gridworld, these would be 4-D non-negative vectors that sum to 1, with each dimension representing the probability of choosing an action.

Crucially, the GVF framework is extremely fast and powerful when the primitives have very low variance. However, zero variance leads to some problems, for instance, getting stuck when the cardinality of the agent's primitives are not aligned with the environment (see Figure in Pablo's slides).

We would design this objective to have no reward such that it can be maximized in an unsupervised manner before any experience. The goal is to set the policies in the correct ballpark, such that even if we later combine the optimization with rewards as and when it is observed, the network can already take full advantage of the GVF framework.

##### Potential solutions
If we were to learn primitives in an unsupervised manner, we want them to have a few key properties:
1. They must be as different from each other as possible
2. They must have low entropy, but not zero

One of the simplest ways to achieve this is to design an objective that maximally separates the primitives while maximizing the entropy of the individual primitives. Let's formalize this.

###### Cross entropy objective
$F_1(\theta)=\sum_{i=1}^K\left[H(\rho_i)+\sum_{j=1}^K D_{KL}(\rho_i||\rho_j)\right]=\sum_i\sum_jH(\rho_i,\rho_j)$,
	where $\theta$ are the parameters of the primitives $\rho_i$. 

This is the most direct translation of the words into well-known information theoretic quantities. One problem, however, is that the entropy is a bounded measure whereas the KL-divergence is unbounded and could go to $\infty$ for non-overlapping distributions. Hence, it would dominate the objective whenever $|\mathcal{A}|\leq K$.

###### Modified cross-entropy with a default primitive
In addition to our $K$ primitive policies, $\rho_1, ..., \rho_K$, let us define a default primitive $\rho_0$ which is a uniform distribution. Now, we could think of a few objective functions:

$F_2(\theta)=\sum_{i>0}\sum_{j>0}H(\rho_i,\rho_j) - \sum_{i>0}D_{KL}\left(\rho_i || \rho_0\right)$

This one minimizes distance from default while being as different from each other as possible. Unfortunately, it has the same issues with the Cross-entropy being unbounded. Alternatively, we could try to maximize distance from the default but that doesn't make so much sense and moreover the policies would end up deterministic.

###### Jensen-Shannon divergence
A better way to do this is with the [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)(JSD). JSD is a measure of distance between distributions that is based on the KL-divergence, but with some notable and useful differences, including that it is symmetric and always has a finite value.

For two distributions $P$ and $Q$, the JSD is defined as:
$JSD(P||Q) = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)$,
	where $M=\frac{1}{2}(P+Q)$

Moreover, it could also be defined over a whole set of distributions as:
$JSD_{w_1, ..., w_K}(\rho_1, ..., \rho_K) = \sum_{k=1}^K D_{KL}(\rho_i||\rho_0) = H(\rho_0) - \sum_{k=1}^K w_k H(\rho_k)$,
	where $\rho_0 = \sum_{k=1}^K w_k \rho_k$
	and $w_k$ are the weights (that sum to 1).

Our objective, would then be:
$F_3(\theta)=JSD(\rho_1, \rho_2, ..., \rho_K) + \alpha\sum_{k=1}^K\left[H(\rho_k)\right]$,
	where $\alpha$ could be a parameter that balances the stochasticity of the policies to their mutual separation.

**Problem**:
This objective also escapes intended stochastic behavior. For instance, in the case of uniform weights, the objective reduces to:
		$F_3(\theta)=H(\rho_0) \pm \lambda\sum_{k=1}^K\left[H(\rho_k)\right]$,
			where 
			$\lambda = -1 \quad\text{if}\quad \alpha=0$,  
				$\lambda = 0 \quad\text{if}\quad \alpha=\frac{1}{K}$, and 
				$\lambda>0 \quad\text{if}\quad \alpha>\frac{1}{K}$ 

Thus, in the best case (when $\alpha=\frac{1}{K}$), it ends up with a set of degenerate solutions.

##### Expected results
For the grid-world with K=1, the solution would be 

#### Motor primitives
Much more complex, real-world behaviors can be learnt by using the framework of motor-primitives. This can be combined with the GVFs in the future with a similar style as we attempt with the unsupervised combination.