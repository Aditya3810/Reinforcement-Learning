# k-armed bandit problem.
A k-armed bandit problem is one of the foremost problems in RL. It depicts the evaluative aspect of reinforcement learning where an evaluative feedback indicates how good the action taken was.
The k-armed bandit is inspired by the slot machines you see in casinos. They have only one lever and are referred to as the one armed bandits. 
In the k-armed bandit, we have more than one lever. Thus having k-arms, and having repeated choice among k-actions. After each choice we receive a reward 
chosen from a stationary distribution that depends on the action chosen.

The goal of the agent is to learn which action gives the highest reward over time.

At each time step:

\[
A_t = \text{action selected at time } t
\]

\[
R_t = \text{reward received at time } t
\]

The challenge is that the agent does not know in advance which action is best.

---

## 2. True Action Value

Each action has a true underlying value.

This is written as:

\[
q_*(a)
\]

It means:

\[
q_*(a) = \mathbb{E}[R_t \mid A_t = a]
\]

In simple words:

> The true action value is the expected reward obtained by choosing action \(a\).

So if an action gives rewards that are noisy, its true value is the long-term average reward we would get from repeatedly selecting that action.

For example:

| Action | True average reward |
|---|---:|
| Arm 1 | 2 |
| Arm 2 | 5 |
| Arm 3 | 1 |

Here, Arm 2 is truly the best action.

However, the agent does not know these true values at the beginning.

---

## 3. Estimated Action Value

Because the agent does not know the true value \(q_*(a)\), it keeps an estimate of it.

This estimate is written as:

\[
Q_t(a)
\]

This means:

> The agent's current estimate of the value of action \(a\) at time \(t\).

So there are two important quantities:

\[
q_*(a) = \text{true value of action } a
\]

\[
Q_t(a) = \text{estimated value of action } a
\]

The learning problem is to make:

\[
Q_t(a) \approx q_*(a)
\]

through experience.

---

## 4. Sample-Average Method

A natural way to estimate the value of an action is to average the rewards received after selecting that action.

\[
Q_t(a) =
\frac{
\text{sum of rewards received when action } a \text{ was taken before time } t
}{
\text{number of times action } a \text{ was taken before time } t
}
\]

In mathematical form:

\[
Q_t(a) =
\frac{
\sum_{i=1}^{t-1} R_i \cdot \mathbf{1}_{A_i = a}
}{
\sum_{i=1}^{t-1} \mathbf{1}_{A_i = a}
}
\]

This looks complicated, but it simply means:

> Look back at all previous time steps before \(t\), find the times when action \(a\) was selected, and average the rewards received from those selections.

---

## 5. What Does “Action Taken Prior to \(t\)” Mean?

“Prior to \(t\)” means before the current time step \(t\).

At time \(t\), the agent has not yet chosen the current action or received the current reward.  
So \(Q_t(a)\) is based only on past experience.

Example:

| Time | Action chosen | Reward |
|---:|---|---:|
| 1 | A | 2 |
| 2 | B | 5 |
| 3 | A | 4 |
| 4 | C | 1 |
| 5 | A | 3 |

Now suppose we are at:

\[
t = 6
\]

To calculate \(Q_6(A)\), we only look at times before 6.

Action A was chosen at times 1, 3, and 5.

Rewards from A were:

\[
2, 4, 3
\]

So:

\[
Q_6(A) = \frac{2 + 4 + 3}{3} = 3
\]

Therefore:

> \(Q_t(a)\) is the estimate available before making the decision at time \(t\).

---

## 6. Indicator Function

The term:

\[
\mathbf{1}_{A_i = a}
\]

is an indicator function.

It means:

\[
\mathbf{1}_{A_i = a} =
\begin{cases}
1, & \text{if action } a \text{ was selected at time } i \\
0, & \text{otherwise}
\end{cases}
\]

So it acts like a filter.

For example, if we want to calculate the value of action A:

| Time \(i\) | Action \(A_i\) | Reward \(R_i\) | \(\mathbf{1}_{A_i=A}\) | \(R_i \cdot \mathbf{1}_{A_i=A}\) |
|---:|---|---:|---:|---:|
| 1 | A | 2 | 1 | 2 |
| 2 | B | 5 | 0 | 0 |
| 3 | A | 4 | 1 | 4 |
| 4 | C | 1 | 0 | 0 |
| 5 | A | 3 | 1 | 3 |

Numerator:

\[
2 + 0 + 4 + 0 + 3 = 9
\]

Denominator:

\[
1 + 0 + 1 + 0 + 1 = 3
\]

So:

\[
Q_6(A) = \frac{9}{3} = 3
\]

---

## 7. What Does “As the Denominator Goes to Infinity” Mean?

The denominator is:

\[
\sum_{i=1}^{t-1} \mathbf{1}_{A_i = a}
\]

This is simply the number of times action \(a\) has been selected before time \(t\).

We can write this as:

\[
N_t(a)
\]

So:

\[
N_t(a) = \text{number of times action } a \text{ has been chosen before time } t
\]

When Sutton & Barto say “as the denominator goes to infinity,” they mean:

\[
N_t(a) \to \infty
\]

In simple words:

> If the agent keeps selecting action \(a\) again and again, then the number of samples for that action becomes very large.

By the law of large numbers, the sample average will eventually approach the true average.

So:

\[
Q_t(a) \to q_*(a)
\]

This means:

> The more often an action is sampled, the more accurate the estimated value of that action becomes.

Example:

Suppose action \(a\) has true value:

\[
q_*(a) = 5
\]

But the actual rewards are noisy:

\[
4, 7, 3, 6, 5, 8, 2, ...
\]

After 2 selections:

\[
Q(a) = \frac{4 + 7}{2} = 5.5
\]

After 10 selections, the estimate may be closer to 5.

After 10,000 selections, the estimate may be very close to:

\[
5
\]

Important point:

> This convergence only happens for actions that are repeatedly sampled.

If an action is selected only once or twice, its estimate may remain inaccurate.

This is why exploration is important.

---

## 8. Greedy Action Selection

Once the agent has estimates \(Q_t(a)\), it can use them to select actions.

The simplest method is greedy action selection:

\[
A_t = \arg\max_a Q_t(a)
\]

This means:

> At time \(t\), choose the action whose current estimated value is highest.

Important distinction:

\[
\max_a Q_t(a)
\]

returns the maximum estimated value.

But:

\[
\arg\max_a Q_t(a)
\]

returns the action that has the maximum estimated value.

Example:

| Action \(a\) | Estimated value \(Q_t(a)\) |
|---|---:|
| Action 1 | 2.1 |
| Action 2 | 4.7 |
| Action 3 | 1.5 |

Here:

\[
\max_a Q_t(a) = 4.7
\]

But:

\[
\arg\max_a Q_t(a) = \text{Action 2}
\]

So greedy action selection chooses Action 2.

In Python:

```python
import numpy as np

Q = [2.1, 4.7, 1.5]

action = np.argmax(Q)

print(action)

