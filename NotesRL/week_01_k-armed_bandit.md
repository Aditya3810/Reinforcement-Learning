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

## 24. Tracking a Non-Stationary Problem

A non-stationary problem means that the reward values change over time.

Earlier, we assumed that the true value of an action stays fixed.

That is called a stationary problem.

In a stationary bandit problem, the true action values do not change.

Example:

| Action | True value at start | True value later |
|---|---:|---:|
| A | 2.0 | 2.0 |
| B | 5.0 | 5.0 |
| C | 1.0 | 1.0 |

The best action remains the same over time.

But in a non-stationary bandit problem, the true values can change.

Example:

| Action | True value at start | True value later |
|---|---:|---:|
| A | 2.0 | 6.0 |
| B | 5.0 | 1.0 |
| C | 1.0 | 3.0 |

At the beginning, action B is the best action.

Later, action A becomes the best action.

So the agent must keep adapting.

---

## 25. Why the Sample-Average Method Becomes a Problem

Earlier, we used the sample-average update:

$$
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
$$

Here, the step size is:

$$
\frac{1}{n}
$$

As \(n\) gets larger, this value becomes very small.

For example:

$$
n = 1000
$$

$$
\frac{1}{n} = 0.001
$$

So new rewards barely change the estimate.

This is useful in a stationary problem because the true action value is fixed.

But in a non-stationary problem, this becomes a problem.

Why?

Because old rewards may no longer be useful.

If an action used to be good but is now bad, the old rewards will keep influencing the estimate too much.

---

## 26. Example: Why Old Rewards Can Be Misleading

Suppose action A used to give rewards around 10.

So after many trials, the agent estimates:

$$
Q(A) = 10
$$

But now the environment changes.

Action A starts giving rewards around 2.

The agent receives:

$$
R = 2
$$

If action A has already been selected 1000 times, then:

$$
Q_{n+1} = 10 + \frac{1}{1000}[2 - 10]
$$

$$
Q_{n+1} = 10 + 0.001[-8]
$$

$$
Q_{n+1} = 10 - 0.008
$$

$$
Q_{n+1} = 9.992
$$

The estimate barely changes.

But the true value has changed from 10 to 2.

So the agent adapts too slowly.

---

## 27. Constant Step-Size Method

To solve this, we replace:

$$
\frac{1}{n}
$$

with a constant value:

$$
\alpha
$$

So the update becomes:

$$
Q_{n+1} = Q_n + \alpha[R_n - Q_n]
$$

Here, \(\alpha\) is called the step size or learning rate.

For example:

$$
\alpha = 0.1
$$

This means that each new reward has a fixed influence on the updated estimate.

The update still follows the same general idea:

$$
\text{New estimate} = \text{Old estimate} + \text{Step size} \times \text{Prediction error}
$$

But now the step size does not shrink over time.

So the agent can keep adapting to new rewards.

---

## 28. Example: Constant Step Size

Suppose:

$$
Q_n = 10
$$

and the new reward is:

$$
R_n = 2
$$

Use:

$$
\alpha = 0.1
$$

Then:

$$
Q_{n+1} = 10 + 0.1[2 - 10]
$$

$$
Q_{n+1} = 10 + 0.1[-8]
$$

$$
Q_{n+1} = 10 - 0.8
$$

$$
Q_{n+1} = 9.2
$$

Now the estimate changes from 10 to 9.2.

This is much faster than 9.992 from the sample-average method.

If the agent keeps receiving low rewards, the estimate will keep moving down.

So a constant step size helps the agent track changes in the environment.

---

## 29. Weighted Average Idea

With a constant step size, the estimate becomes a weighted average of past rewards.

The expanded form is:

$$
Q_{n+1} = (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i}R_i
$$

This looks complicated, but the idea is simple:

The estimate depends on:

- the initial estimate \(Q_1\)
- all previous rewards
- how recent those rewards are

Recent rewards get more weight.

Older rewards get less weight.

That is why this method is useful for non-stationary problems.

---

## 30. Meaning of the Weight

The weight given to reward \(R_i\) is:

$$
\alpha(1-\alpha)^{n-i}
$$

Here:

$$
n-i
$$

means how long ago the reward happened.

If \(R_i\) happened recently, then \(n-i\) is small.

So the weight is larger.

If \(R_i\) happened long ago, then \(n-i\) is large.

So the weight becomes smaller.

Because:

$$
0 < 1-\alpha < 1
$$

raising it to larger powers makes it smaller.

---

## 31. Example with \(\alpha = 0.1\)

Let:

$$
\alpha = 0.1
$$

Then:

$$
1-\alpha = 0.9
$$

The most recent reward gets weight:

$$
0.1
$$

The reward before that gets weight:

$$
0.1 \times 0.9 = 0.09
$$

The reward before that gets weight:

$$
0.1 \times 0.9^2 = 0.081
$$

The reward before that gets weight:

$$
0.1 \times 0.9^3 = 0.0729
$$

So the weights look like this:

| Reward | Weight |
|---|---:|
| Most recent reward | 0.1000 |
| 1 step older | 0.0900 |
| 2 steps older | 0.0810 |
| 3 steps older | 0.0729 |
| 4 steps older | 0.0656 |

Older rewards are not ignored completely, but their influence fades.

This is called an exponential recency-weighted average.

In simple words:

The agent remembers the past, but it trusts recent experience more.

---

## 32. Why the Weights Sum to 1

The estimate is still an average.

But it is not a normal average where every reward has equal weight.

In a normal sample average:

$$
Q = \frac{R_1 + R_2 + R_3 + R_4}{4}
$$

Each reward has equal weight:

$$
0.25
$$

But in the constant step-size method, recent rewards get larger weights and older rewards get smaller weights.

So the estimate is biased toward recent experience.

This is exactly what we want when the environment changes.

---

## 33. Optimistic Initial Values

Optimistic initial values are a way to encourage exploration.

Usually, we initialize action-value estimates like this:

$$
Q_1(a) = 0
$$

for all actions.

But in optimistic initial values, we start with unrealistically high estimates.

For example:

$$
Q_1(a) = 5
$$

for all actions.

This means the agent initially believes every action is very good.

---

## 34. Why Optimistic Initial Values Encourage Exploration

Suppose we have 4 actions.

Initial estimates:

| Action | Initial estimate |
|---|---:|
| A | 5 |
| B | 5 |
| C | 5 |
| D | 5 |

Now the agent tries action A and gets reward:

$$
R = 1
$$

The estimate for A decreases.

For example:

| Action | Estimate |
|---|---:|
| A | 3 |
| B | 5 |
| C | 5 |
| D | 5 |

Now the greedy agent chooses one of B, C, or D because they still look better.

Then it tries B.

If B gives a low reward, B also decreases.

Eventually, the agent is pushed to try all actions because unexplored actions still have high optimistic values.

So even a greedy agent is encouraged to explore.

---

## 35. Example of Optimistic Initial Values

Suppose the true values are:

| Action | True value |
|---|---:|
| A | 1 |
| B | 2 |
| C | 4 |

But the agent starts with optimistic estimates:

| Action | Initial estimate |
|---|---:|
| A | 10 |
| B | 10 |
| C | 10 |

The agent chooses A first.

Suppose the reward is:

$$
R = 1
$$

A's estimate drops.

Now B and C still look better because their estimates are still high.

So the agent tries B or C next.

This forces exploration.

After enough experience, the estimates move closer to the true values:

| Action | Estimated value |
|---|---:|
| A | 1 |
| B | 2 |
| C | 4 |

Then the agent mostly chooses C.

---

## 36. Why Is It Called Optimistic?

It is called optimistic because the agent starts with overly positive beliefs.

It assumes every action is better than it probably is.

As the agent samples actions, reality corrects those estimates.

This is useful because unexplored actions remain attractive until they are tried.

---

## 37. Limitation of Optimistic Initial Values

Optimistic initial values are useful mainly in stationary problems.

In a stationary problem, the true values do not change.

Once the agent has explored and corrected its estimates, it can exploit the best action.

But in a non-stationary problem, optimistic initial values only encourage exploration at the beginning.

Later, if the environment changes, the initial optimism is gone.

So it does not keep encouraging exploration forever.

For non-stationary problems, methods like epsilon-greedy action selection or constant step-size updates are usually more useful.

---

## 38. Summary in My Own Words

A non-stationary problem means that the reward values change over time.

In this case, old rewards may become less useful.

The sample-average update:

$$
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
$$

learns too slowly after many trials because:

$$
\frac{1}{n}
$$

becomes very small.

So we use a constant step size:

$$
Q_{n+1} = Q_n + \alpha[R_n - Q_n]
$$

This gives more weight to recent rewards and less weight to older rewards.

Optimistic initial values are a different way to encourage exploration.

They start all action-value estimates at high values, making unexplored actions look attractive.

In short:

Constant step size helps the agent adapt to change.

Optimistic initial values help the agent explore early.


## 42. Upper-Confidence-Bound Action Selection

Upper-Confidence-Bound action selection is another way to handle the exploration problem.

Earlier, we saw epsilon-greedy action selection.

In epsilon-greedy action selection, the agent usually chooses the best-looking action, but sometimes chooses a random action.

The problem is that random exploration may waste time.

For example, if some actions already look very bad, epsilon-greedy may still choose them randomly during exploration.

Upper-Confidence-Bound action selection tries to explore more intelligently.

Instead of asking only:

Which action currently looks best?

UCB asks:

Which action has the best combination of high estimated value and high uncertainty?

---

## 43. Main Idea Behind UCB

UCB chooses actions based on two things:

1. how good the action currently looks
2. how uncertain the agent is about that action

So UCB does not only use the estimated action value:

$$
Q_t(a)
$$

It also adds an uncertainty bonus.

The agent then chooses the action with the highest combined score.

In simple words:

The agent chooses actions that either look good already, or have not been explored enough and might still turn out to be good.

---

## 44. UCB Formula

The UCB action selection formula is:

$$
A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

This means:

At time \(t\), choose the action \(a\) that maximizes:

$$
Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}}
$$

The first part is:

$$
Q_t(a)
$$

This is the current estimated value of action \(a\).

The second part is:

$$
c \sqrt{\frac{\ln t}{N_t(a)}}
$$

This is the uncertainty bonus.

So the full score is:

$$
\text{UCB score} = \text{estimated value} + \text{uncertainty bonus}
$$

---

## 45. Meaning of \(Q_t(a)\)

The term:

$$
Q_t(a)
$$

is the agent's current estimate of the value of action \(a\).

It tells us how good the action looks based on previous rewards.

If \(Q_t(a)\) is high, the action currently looks promising.

This part encourages exploitation.

Exploitation means choosing actions that already seem good.

---

## 46. Meaning of the Uncertainty Bonus

The term:

$$
c \sqrt{\frac{\ln t}{N_t(a)}}
$$

is the uncertainty bonus.

This part encourages exploration.

It gives extra value to actions that have not been selected very often.

If an action has been selected only a few times, then the agent is still uncertain about its true value.

So UCB gives that action a larger bonus.

If an action has been selected many times, then the agent is more confident about its value.

So UCB gives that action a smaller bonus.

---

## 47. Meaning of \(t\)

The term:

$$
t
$$

is the current time step.

As the experiment continues:

$$
t = 1, 2, 3, 4, \dots
$$

So \(t\) increases at every time step.

The term:

$$
\ln t
$$

also increases over time, but slowly.

This means that as time passes, the algorithm maintains some pressure to explore.

---

## 48. Meaning of \(N_t(a)\)

The term:

$$
N_t(a)
$$

means the number of times action \(a\) has been selected before time \(t\).

If action \(a\) has been selected many times, then:

$$
N_t(a)
$$

is large.

So the uncertainty bonus becomes smaller.

If action \(a\) has been selected only a few times, then:

$$
N_t(a)
$$

is small.

So the uncertainty bonus becomes larger.

This encourages the agent to try actions it has not explored much.

---

## 49. Meaning of \(c\)

The term:

$$
c
$$

controls the amount of exploration.

It is called the confidence level or exploration parameter.

Usually:

$$
c > 0
$$

If \(c\) is large, the uncertainty bonus becomes larger.

So the agent explores more.

If \(c\) is small, the uncertainty bonus becomes smaller.

So the agent behaves more greedily.

---

## 50. Simple UCB Example

Suppose there are 3 actions at time:

$$
t = 100
$$

Let:

$$
c = 2
$$

Current estimates:

| Action | \(Q_t(a)\) | \(N_t(a)\) |
|---|---:|---:|
| A | 5.0 | 50 |
| B | 4.8 | 5 |
| C | 3.0 | 45 |

Action A has the highest estimated value:

$$
Q_t(A) = 5.0
$$

So greedy action selection would choose action A.

But UCB also considers uncertainty.

Action B has been selected only 5 times.

So the agent is still uncertain about action B.

Because \(N_t(B)\) is small, action B gets a larger uncertainty bonus.

So even though action B has a slightly lower estimate than action A, UCB may still choose action B.

This is because action B still has potential to be optimal.

---

## 51. What Happens When an Action Is Selected?

Suppose action \(a\) is selected.

Then:

$$
N_t(a)
$$

increases.

Since \(N_t(a)\) appears in the denominator:

$$
\sqrt{\frac{\ln t}{N_t(a)}}
$$

the uncertainty bonus decreases.

So the more often an action is selected, the less uncertain the agent becomes about that action.

In simple words:

The more I try an action, the less curious I am about it.

---

## 52. What Happens When an Action Is Not Selected?

If an action is not selected, then \(N_t(a)\) does not increase.

But time still increases.

So:

$$
\ln t
$$

increases.

This means the uncertainty bonus for ignored actions can slowly grow over time.

So if an action has not been selected for a long time, it can become attractive again.

In simple words:

If I have ignored an action for a long time, maybe I should check it again.

---

## 53. UCB Intuition

UCB is like saying:

I will choose the action that either looks good already, or has not been tested enough and might still turn out to be good.

So UCB does not explore randomly like epsilon-greedy.

Instead, it explores where uncertainty is high.

This makes exploration more directed.

---

## 54. Gradient Bandit Algorithms

Gradient bandit algorithms are another way to select actions.

Until now, we estimated action values:

$$
Q_t(a)
$$

and used those estimates to select actions.

Gradient bandit algorithms do something different.

They do not directly estimate action values.

Instead, they learn a numerical preference for each action.

This preference is written as:

$$
H_t(a)
$$

---

## 55. What Is \(H_t(a)\)?

The term:

$$
H_t(a)
$$

means the preference for action \(a\) at time \(t\).

A higher preference means the action is more likely to be selected.

A lower preference means the action is less likely to be selected.

Important distinction:

$$
Q_t(a)
$$

means estimated reward value.

But:

$$
H_t(a)
$$

means preference or tendency to choose that action.

So \(H_t(a)\) is not the same as \(Q_t(a)\).

---

## 56. Action Selection Using Softmax

Gradient bandit algorithms convert action preferences into probabilities.

This is done using softmax.

The probability of selecting action \(a\) is:

$$
\pi_t(a) =
\frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}
$$

This means each action gets a probability.

The higher the preference \(H_t(a)\), the higher the probability of choosing that action.

So the agent does not always choose the maximum preference action.

Instead, it samples actions according to probabilities.

This naturally allows exploration.

---

## 57. Simple Softmax Example

Suppose there are 3 actions:

| Action | Preference \(H_t(a)\) |
|---|---:|
| A | 2.0 |
| B | 1.0 |
| C | 0.0 |

Action A has the highest preference.

So action A will have the highest probability of being selected.

But actions B and C can still be selected sometimes.

This means softmax does not completely ignore lower-preference actions.

So it gives a natural balance between exploration and exploitation.

---

## 58. Updating Action Preferences

After selecting action \(A_t\) and receiving reward \(R_t\), the preference of the selected action is updated.

For the selected action:

$$
H_{t+1}(A_t) = H_t(A_t) + \alpha(R_t - \bar{R}_t)(1 - \pi_t(A_t))
$$

For all actions that were not selected:

$$
H_{t+1}(a) = H_t(a) - \alpha(R_t - \bar{R}_t)\pi_t(a)
$$

where:

$$
a \neq A_t
$$

Note:

The step size \(\alpha\) is usually positive:

$$
\alpha > 0
$$

The non-selected action update usually has a minus sign.

---

## 59. Meaning of \(\bar{R}_t\)

The term:

$$
\bar{R}_t
$$

is the average reward received so far.

It is used as a baseline.

The term:

$$
R_t - \bar{R}_t
$$

means:

Was the reward better or worse than usual?

If:

$$
R_t > \bar{R}_t
$$

then the reward was better than average.

If:

$$
R_t < \bar{R}_t
$$

then the reward was worse than average.

So the agent does not only ask:

Did I get a reward?

It asks:

Was this reward better than what I usually get?

---

## 60. Why Use a Baseline?

Suppose the agent receives reward:

$$
R_t = 5
$$

Is 5 good?

It depends on the average reward.

If the average reward is:

$$
\bar{R}_t = 2
$$

then:

$$
R_t - \bar{R}_t = 5 - 2 = 3
$$

So the reward is better than usual.

The selected action should become more preferred.

But if the average reward is:

$$
\bar{R}_t = 8
$$

then:

$$
R_t - \bar{R}_t = 5 - 8 = -3
$$

So the reward is worse than usual.

The selected action should become less preferred.

This is why the baseline matters.

---

## 61. Example: Reward Above Baseline

Suppose action A was selected.

Reward:

$$
R_t = 10
$$

Average reward so far:

$$
\bar{R}_t = 6
$$

So:

$$
R_t - \bar{R}_t = 4
$$

The reward is better than usual.

So the preference for action A increases.

$$
H(A) \uparrow
$$

The probabilities of the non-selected actions decrease slightly.

In simple words:

That action gave me a better-than-usual reward, so I should choose it more often.

---

## 62. Example: Reward Below Baseline

Suppose action A was selected.

Reward:

$$
R_t = 3
$$

Average reward so far:

$$
\bar{R}_t = 6
$$

So:

$$
R_t - \bar{R}_t = -3
$$

The reward is worse than usual.

So the preference for action A decreases.

$$
H(A) \downarrow
$$

The probabilities of the non-selected actions increase slightly.

In simple words:

That action gave me a worse-than-usual reward, so I should choose it less often.

---

## 63. Why Do Non-Selected Actions Change?

In gradient bandit algorithms, even the actions that were not selected are updated.

This can feel strange at first.

But the reason is that action probabilities must balance.

The softmax probabilities must sum to 1.

If the probability of one action increases, the probabilities of the other actions must decrease.

If the probability of the selected action decreases, the probabilities of the other actions can increase.

So the non-selected actions move in the opposite direction.

---

## 64. Meaning of \(1 - \pi_t(A_t)\)

For the selected action, the update contains:

$$
1 - \pi_t(A_t)
$$

This controls how much the selected action's preference changes.

If the selected action already had a very high probability, then:

$$
\pi_t(A_t)
$$

is close to 1.

So:

$$
1 - \pi_t(A_t)
$$

is small.

This means the update is smaller.

If the selected action had a low probability but gave a good reward, then:

$$
1 - \pi_t(A_t)
$$

is large.

This means the update is larger.

In simple words:

If a surprising action gives a good reward, increase its preference strongly.

---

## 65. Meaning of \(\pi_t(a)\) for Non-Selected Actions

For non-selected actions, the update uses:

$$
\pi_t(a)
$$

Actions that already had high probability are adjusted more.

Actions that had low probability are adjusted less.

This helps redistribute probability smoothly across all actions.

---

## 66. Difference Between UCB and Gradient Bandit Methods

UCB uses action-value estimates:

$$
Q_t(a)
$$

and an uncertainty bonus:

$$
c \sqrt{\frac{\ln t}{N_t(a)}}
$$

It chooses the action with the highest upper-confidence score.

So UCB asks:

Which action looks best, considering both value and uncertainty?

Gradient bandit algorithms do not directly estimate action values.

They learn preferences:

$$
H_t(a)
$$

Then these preferences are converted into probabilities:

$$
\pi_t(a)
$$

using softmax.

So gradient bandit algorithms ask:

How should I change my probability of selecting each action based on whether the reward was better or worse than usual?

---

## 67. Summary in My Own Words

Upper-Confidence-Bound action selection improves exploration by choosing actions that either have high estimated value or high uncertainty.

Its formula is:

$$
A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

The first part:

$$
Q_t(a)
$$

encourages exploitation.

The second part:

$$
c \sqrt{\frac{\ln t}{N_t(a)}}
$$

encourages exploration.

Gradient bandit algorithms use preferences instead of value estimates.

Each action has a preference:

$$
H_t(a)
$$

These preferences are converted into action probabilities using softmax:

$$
\pi_t(a)
$$

After receiving a reward, the selected action's preference is increased if the reward is above the average baseline.

The selected action's preference is decreased if the reward is below the average baseline.

In short:

UCB explores based on uncertainty.

Gradient bandit algorithms learn which actions should become more or less probable based on reward compared to a baseline.