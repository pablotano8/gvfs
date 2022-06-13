[[Research projects MOC]]


# Compositional GVFs
## Background
Pablo has developed a way to ask arbitrary questions to which easy answers can be generated in a compositional manner by using Generalized Value Functions (GVFs) with simple policies. The key is that if you use low variance policies such as always going straight, learning is incredibly fast. Basically, a network has to learn to map the features (GVFs) to policies, which turns out to be a straightforward problem, because the mapping is more or less one-to-one: if a level 0 GVF is high, take the action corresponding to the GVF to reach the goal, else, if a level 1 GVF is high, do the same to reach a state from where you can reach the goal (simply), and so on until the lowest level. 

There seem to be at least one problem with this approach, which is that the primitive policies need to be hand-defined. For now, it is simply assumed that the policy set is the same as the action set, i.e. up, down, left, right. In general, you would like to learn arbitrarily complex policies that make up your set of primitive policies.

### If sub-goals are given
If the sub-goals are specified, how would you solve the problem quickly using the framework? One idea is to construct a hierarchical level saying that at the top level, your set of primitive policies are the policies that take you to the respective sub-goals. Let's consider an environment where you need to fetch two keys, k1 and k2, and then head to the goal. Thus, your higher-level policy set is $\Pi_{a} \in \{\pi_{k1}, \pi_{k2}, \pi_{g}\}$, and your lower level policy set is $\Pi \in \{\uparrow, \downarrow, \rightarrow, \leftarrow\}$. Initially, the higher-level policies are unknown, and can be random. If the sub-goals are treated as goals, then the problem is reduced to 

## Learning primitive policies
As powerful as the framework is, the story would sell much better if we could demonstrate an easy way to learn the primitive policies, at least in simple cases. Given the pairing with the framework, the goal is to find ==low entropy primitives that are different from each other==. Ideally, we would like to have an unsupervised approach to do this.

[[CGVFs learning primitives]]


## Other links
[[Compositional GVFs abstract]]