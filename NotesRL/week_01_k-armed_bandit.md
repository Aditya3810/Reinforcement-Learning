# k-armed bandit problem.
A k-armed bandit problem is one of the foremost problems in RL. It depicts the evaluative aspect of reinforcement learning where an evaluative feedback indicates how good the action taken was.
The k-armed bandit is inspired by the slot machines you see in casinos. They have only one lever and are referred to as the one armed bandits. 
In the k-armed bandit, we have more than one lever. Thus having k-arms, and having repeated choice among k-actions. After each choice we receive a reward 
chosen from a stationary distribution that depends on the action chosen.

## Objective 
The objective here is to maximize the rewards over some time period.


![alt text](image-1.png)

## Action-Value methods.

These are methods used to calculate the estimates Q<sub>a</sub>(t) for our action at time t.
One natural way to do that is to average the rewards actually received (sample average).

$$
Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = a}}
{\sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}
$$

which means:

$$
Q_t(a) =
\frac{
\text{total reward received from action } a \text{ before time } t
}{
\text{number of times action } a \text{ was selected before time } t
}
$$

## Indicator Function Example

The indicator function is written as:

$$
\mathbb{1}_{A_i = a}
$$

This means:

$$
\mathbb{1}_{A_i = a} =
\begin{cases}
1, & \text{if } A_i = a \\
0, & \text{if } A_i \neq a
\end{cases}
$$

In simple words, the indicator function checks whether the action taken at time $i$ was action $a$.

If the action was $a$, the indicator becomes $1$.

If the action was not $a$, the indicator becomes $0$.

So it filters only the rewards from the action we care about.

---

## Example

Suppose we are interested in action $A$.

| Time $(i)$ | Action $(A_i)$ | Reward $(R_i)$ | $\mathbb{1}_{A_i = A}$ | $R_i \cdot \mathbb{1}_{A_i = A}$ |
|---|---|---:|---:|---:|
| 1 | A | 2 | 1 | 2 |
| 2 | B | 5 | 0 | 0 |
| 3 | A | 4 | 1 | 4 |
| 4 | C | 1 | 0 | 0 |
| 5 | A | 3 | 1 | 3 |

The indicator function keeps only the rewards from action $A$.

So the rewards from action $A$ are:

$$
2, 4, 3
$$

The rewards from other actions are ignored because the indicator becomes $0$.

---

## Numerator

The numerator is the total reward received from action $A$ before time $t$:

$$
2 + 0 + 4 + 0 + 3 = 9
$$

So:

$$
\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = A} = 9
$$

---

## Denominator

The denominator is the number of times action $A$ was selected before time $t$:

$$
1 + 0 + 1 + 0 + 1 = 3
$$

So:

$$
\sum_{i=1}^{t-1} \mathbb{1}_{A_i = A} = 3
$$

---

## Action-value estimate

The action-value estimate is:

$$
Q_t(A) =
\frac{
\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i = A}
}{
\sum_{i=1}^{t-1} \mathbb{1}_{A_i = A}
}
$$

Substituting the values:

$$
Q_t(A) = \frac{9}{3}
$$

Therefore:

$$
Q_t(A) = 3
$$

---

## Intuition

$Q_t(A)$ is the estimated value of action $A$ at time $t$.

It is calculated as:

$$
Q_t(A) =
\frac{
\text{total reward received from action } A \text{ before time } t
}{
\text{number of times action } A \text{ was selected before time } t
}
$$

So in simple words:

> The value of an action is the average reward received from choosing that action so far.

For this example, action $A$ gave rewards $2$, $4$, and $3$.

So the estimated value of action $A$ is:

$$
Q_t(A) = \frac{2 + 4 + 3}{3} = 3
$$

# Greedy and Epsilon-Greedy Action Selection

## Argmax in action selection


$$
\arg\max_a Q_t(a)
$$

does **not** return the maximum value itself.

It returns the **action** $a$ whose estimated value $Q_t(a)$ is the highest.

---

## Example

| Action $a$ | Estimated value $Q_t(a)$ |
|---|---:|
| Action 1 | 2.1 |
| Action 2 | 4.7 |
| Action 3 | 1.5 |

Here:

$$
\max_a Q_t(a) = 4.7
$$

That is the **maximum estimated value**.

But:

$$
\arg\max_a Q_t(a) = \text{Action 2}
$$

That is the **action that gives the maximum value**.

So this:

$$
A_t = \arg\max_a Q_t(a)
$$

means:

> At time $t$, choose the action whose current estimated value is highest.

---

# 9. Problem With Pure Greedy Selection

Greedy action selection always chooses the action that currently looks best.

This can be a problem early in learning.

For example, suppose the agent tries Action 1 once and gets a high reward by chance.

It may then keep choosing Action 1 forever, even if another action is actually better.

So pure greedy selection can get stuck because it does not explore enough.

This is called the **exploration-exploitation tradeoff**.

---

# 10. Epsilon-Greedy Action Selection

An alternative is **epsilon-greedy action selection**.

In epsilon-greedy selection, the agent usually behaves greedily, but sometimes explores.

With probability:

$$
1 - \epsilon
$$

the agent chooses the greedy action:

$$
A_t = \arg\max_a Q_t(a)
$$

With probability:

$$
\epsilon
$$

the agent chooses a random action.

---

## In simple words

> Most of the time, choose the action that currently looks best.  
> Sometimes, choose randomly to explore other possibilities.

---

## Example

If:

$$
\epsilon = 0.1
$$

then:

- 90% of the time, the agent exploits the best-known action.
- 10% of the time, the agent explores randomly.

So if the current estimated values are:

| Action $a$ | Estimated value $Q_t(a)$ |
|---|---:|
| Action 1 | 2.1 |
| Action 2 | 4.7 |
| Action 3 | 1.5 |

The greedy action is:

$$
\arg\max_a Q_t(a) = \text{Action 2}
$$

With epsilon-greedy action selection:

- Most of the time, the agent chooses Action 2.
- Sometimes, it randomly chooses Action 1, Action 2, or Action 3.

---

## Why exploration matters

Exploration is important because the agent does not know the true value of each action at the beginning.

An action that looks bad early may actually be good.

An action that looks good early may only look good because of lucky rewards.

So epsilon-greedy selection prevents the agent from blindly trusting early estimates.

---

## Summary

Pure greedy selection:

$$
A_t = \arg\max_a Q_t(a)
$$

always chooses the action with the highest estimated value.

Epsilon-greedy selection:

$$
A_t =
\begin{cases}
\arg\max_a Q_t(a), & \text{with probability } 1 - \epsilon \\
\text{random action}, & \text{with probability } \epsilon
\end{cases}
$$

This balances two things:

| Term | Meaning |
|---|---|
| Exploitation | Choosing the best-known action |
| Exploration | Trying other actions to learn more |

So epsilon-greedy means:

> Exploit most of the time, but explore sometimes.