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

\[
\mathbf{1}_{A_i = a}
\]

It equals:

\[
\mathbf{1}_{A_i = a} =
\begin{cases}
1, & \text{if } A_i = a \\
0, & \text{if } A_i \neq a
\end{cases}
\]

So it filters only the rewards from the action we care about.

For example, for action \(A\):

| Time \(i\) | Action \(A_i\) | Reward \(R_i\) | \(\mathbf{1}_{A_i = A}\) | \(R_i \cdot \mathbf{1}_{A_i = A}\) |
|---:|:---:|---:|---:|---:|
| 1 | A | 2 | 1 | 2 |
| 2 | B | 5 | 0 | 0 |
| 3 | A | 4 | 1 | 4 |
| 4 | C | 1 | 0 | 0 |
| 5 | A | 3 | 1 | 3 |

The numerator is the total reward received from action \(A\):

\[
2 + 0 + 4 + 0 + 3 = 9
\]

The denominator is the number of times action \(A\) was selected:

\[
1 + 0 + 1 + 0 + 1 = 3
\]

Therefore:

\[
Q_6(A) = \frac{9}{3} = 3
\]

So the estimated value of action \(A\) before time step 6 is:

\[
Q_6(A) = 3
\]