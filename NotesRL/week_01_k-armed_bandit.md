# k-armed bandit problem.
A k-armed bandit problem is one of the foremost problems in RL. It depicts the evaluative aspect of reinforcement learning where an evaluative feedback indicates how good the action taken was.
The k-armed bandit is inspired by the slot machines you see in casinos. They have only one lever and are referred to as the one armed bandits. 
In the k-armed bandit, we have more than one lever. Thus having k-arms, and having repeated choice among k-actions. After each choice we receive a reward 
chosen from a stationary distribution that depends on the action chosen.

## 1. Objective 
The objective here is to maximize the rewards over some time period.


![alt text](image-1.png)

## 2. Action-Value methods.

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

## 3. Indicator Function Example

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

## 4. Action-value estimate

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

## 5. Intuition

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

# 6. Greedy and Epsilon-Greedy Action Selection

## 7. Argmax in action selection


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

# 8. Problem With Pure Greedy Selection

Greedy action selection always chooses the action that currently looks best.

This can be a problem early in learning.

For example, suppose the agent tries Action 1 once and gets a high reward by chance.

It may then keep choosing Action 1 forever, even if another action is actually better.

So pure greedy selection can get stuck because it does not explore enough.

This is called the **exploration-exploitation tradeoff**.

---

# 9. Epsilon-Greedy Action Selection

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

## 10. Why exploration matters

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

## 11. The 10-Armed Testbed

A 10-armed bandit means that the agent has 10 possible actions.

You can think of this as 10 slot machines.

At each time step, the agent chooses one action and receives a reward.

Each action has a true value:

$$
q_*(a)
$$

This true value is unknown to the agent.

The agent has to learn which action is best by selecting actions and observing rewards.

---

## 12. How the Testbed Is Created

In Sutton and Barto's 10-armed testbed, each action's true value is selected from a Gaussian distribution.

$$
q_*(a) \sim \mathcal{N}(0,1)
$$

This means that the true value of each action is randomly sampled from a normal distribution with:

- mean = 0
- variance = 1

For example, one 10-armed bandit problem may have these true values:

| Action | True value |
|---:|---:|
| 1 | 0.3 |
| 2 | -1.1 |
| 3 | 1.7 |
| 4 | 0.8 |
| 5 | -0.4 |
| 6 | 2.1 |
| 7 | 0.0 |
| 8 | -0.7 |
| 9 | 1.2 |
| 10 | 0.5 |

Here, action 6 is the best action because it has the highest true value.

$$
q_*(6) = 2.1
$$

But the agent does not know this in the beginning.

It has to learn this through experience.

---

## 13. Reward Generation

When the agent selects an action, the reward is also sampled from a Gaussian distribution.

If the agent selects action \(A_t\), then the reward is sampled as:

$$
R_t \sim \mathcal{N}(q_*(A_t), 1)
$$

This means that the reward is centered around the true value of the selected action, but it is noisy.

For example, suppose action 6 has true value:

$$
q_*(6) = 2.1
$$

If the agent selects action 6 multiple times, the rewards may look like:

$$
1.8,\ 2.5,\ 0.9,\ 3.1,\ 2.0,\ 1.4
$$

The reward is not always exactly 2.1.

But over many selections, the average reward will move closer to 2.1.

This is why the true action value is an expected reward, not a fixed reward.

---

## 14. Why Rewards Are Noisy

Even if an action has a high true value, it can sometimes give a low reward.

Similarly, even if an action has a low true value, it can sometimes give a high reward.

Example:

| Action | True value | Possible reward |
|---:|---:|---:|
| Action 1 | 2.1 | 0.9 |
| Action 2 | -0.4 | 1.5 |

Here, Action 2 gives a better reward on one trial by chance.

But over many trials, Action 1 is better on average.

So the agent should not judge an action based on only one reward.

It needs repeated experience to form a reliable estimate.

---

## 15. Why Use Many Bandit Problems?

Sutton and Barto test the learning algorithm on many independent bandit problems.

For example, they use 2000 different 10-armed bandit problems.

Why?

Because one run can be lucky or unlucky.

A greedy agent may perform well in one run simply because it got lucky early.

So we average performance across many runs.

This gives a more reliable comparison between methods.

The general result is:

$$
\epsilon\text{-greedy} > \text{greedy}
$$

This means epsilon-greedy usually performs better than pure greedy because it continues to explore.

---

## 16. Incremental Implementation

To estimate the value of an action, we could store all the rewards and compute their average.

For a single action, suppose the rewards are:

$$
R_1,\ R_2,\ R_3,\ \dots,\ R_{n-1}
$$

The estimate after \(n-1\) selections is:

$$
Q_n = \frac{R_1 + R_2 + \dots + R_{n-1}}{n-1}
$$

This is the sample average.

But storing all previous rewards is inefficient.

Instead, we use an incremental update rule.

This allows us to update the estimate using only:

- the old estimate
- the new reward
- the number of times the action has been selected

---

## 17. Why Not Recompute the Average Every Time?

Suppose action A has produced these rewards:

$$
2,\ 4,\ 3,\ 5
$$

The current estimate is:

$$
Q = \frac{2 + 4 + 3 + 5}{4}
$$

$$
Q = \frac{14}{4} = 3.5
$$

Now suppose the agent selects action A again and receives reward:

$$
R = 6
$$

The new average would be:

$$
Q = \frac{2 + 4 + 3 + 5 + 6}{5}
$$

$$
Q = \frac{20}{5} = 4.0
$$

This works, but it requires storing all previous rewards.

Instead, we update the estimate directly from the previous estimate.

---

## 18. Incremental Update Rule

The incremental update rule is:

$$
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
$$

This means:

$$
\text{New estimate} = \text{Old estimate} + \text{Step size} \times \text{Error}
$$

The general form is:

$$
NewEstimate \leftarrow OldEstimate + StepSize[Target - OldEstimate]
$$

For the bandit problem:

| General term | Bandit meaning |
|---|---|
| OldEstimate | current estimate \(Q_n\) |
| Target | new reward \(R_n\) |
| StepSize | \(\frac{1}{n}\) |
| Target - OldEstimate | prediction error |

So:

$$
R_n - Q_n
$$

is the error in the current estimate.

It tells us how different the received reward was from what the agent expected.

---

## 19. Prediction Error

The term:

$$
R_n - Q_n
$$

is the prediction error.

It means:

$$
\text{Prediction error} = \text{Received reward} - \text{Expected reward}
$$

If:

$$
R_n > Q_n
$$

then the reward was higher than expected, so the estimate should increase.

If:

$$
R_n < Q_n
$$

then the reward was lower than expected, so the estimate should decrease.

If:

$$
R_n = Q_n
$$

then the reward matched the estimate, so the estimate does not change.

---

## 20. Example: Updating After a New Reward

Suppose action A has already been selected 4 times.

The rewards were:

$$
2,\ 4,\ 3,\ 5
$$

The current estimate is:

$$
Q_5 = 3.5
$$

Now the agent selects action A again and receives:

$$
R_5 = 6
$$

Since this is the 5th reward for this action:

$$
n = 5
$$

Using the incremental update rule:

$$
Q_6 = Q_5 + \frac{1}{5}[R_5 - Q_5]
$$

Substitute the values:

$$
Q_6 = 3.5 + \frac{1}{5}[6 - 3.5]
$$

$$
Q_6 = 3.5 + \frac{1}{5}[2.5]
$$

$$
Q_6 = 3.5 + 0.5
$$

$$
Q_6 = 4.0
$$

So the estimate moves from 3.5 to 4.0.

This gives the same answer as calculating the full average:

$$
Q = \frac{2 + 4 + 3 + 5 + 6}{5}
$$

$$
Q = \frac{20}{5} = 4.0
$$

So the incremental update gives the same answer without storing all previous rewards.

---

## 21. Example: Reward Lower Than Expected

Suppose the current estimate is:

$$
Q_n = 5
$$

The agent receives reward:

$$
R_n = 2
$$

The prediction error is:

$$
R_n - Q_n = 2 - 5 = -3
$$

The reward was lower than expected.

If:

$$
n = 10
$$

then:

$$
Q_{n+1} = 5 + \frac{1}{10}[-3]
$$

$$
Q_{n+1} = 5 - 0.3
$$

$$
Q_{n+1} = 4.7
$$

So the estimate decreases from 5 to 4.7.

In simple words:

I expected around 5, but I received 2, so I reduce my estimate slightly.

---

## 22. Example: Reward Higher Than Expected

Suppose the current estimate is:

$$
Q_n = 5
$$

The new reward is:

$$
R_n = 8
$$

The prediction error is:

$$
R_n - Q_n = 8 - 5 = 3
$$

The reward was higher than expected.

If:

$$
n = 10
$$

then:

$$
Q_{n+1} = 5 + \frac{1}{10}[3]
$$

$$
Q_{n+1} = 5 + 0.3
$$

$$
Q_{n+1} = 5.3
$$

So the estimate increases from 5 to 5.3.

In simple words:

I expected around 5, but I received 8, so I increase my estimate slightly.

---

## 23. Meaning of the Step Size

In the sample-average method, the step size is:

$$
\frac{1}{n}
$$

This controls how much the new reward changes the estimate.

Early in learning, \(n\) is small.

For example:

$$
n = 1
$$

$$
\frac{1}{n} = 1
$$

So the first reward has a large effect on the estimate.

Later in learning, \(n\) is large.

For example:

$$
n = 100
$$

$$
\frac{1}{n} = 0.01
$$

So one new reward changes the estimate only slightly.

This makes sense because after many observations, one new reward should not drastically change the estimate.

---

## 24. Code Version of the Incremental Update

In Python, the incremental update looks like this:

```python
N[action] += 1
Q[action] = Q[action] + (1 / N[action]) * (reward - Q[action])