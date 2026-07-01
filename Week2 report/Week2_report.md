# Week 2 Report: Deep Reinforcement Learning for Electric Vehicle Routing with Time Windows (EVRP‑TW)

## 1. Literature Review

Two recent publications were reviewed to identify the most suitable algorithmic foundation for this research.

### 1.1 Bi et al. (2024)
**Bi, X., Wang, R. and Jia, Q. (2024)** *'On the speed‑varying range of electric vehicles in time‑windowed routing problems with en‑route partial re‑charging'*, IEEE Transactions on Consumer Electronics, 70(1), pp. 3650–3657.

#### Main Contributions
*   **Energy Consumption Model**: Employs a longitudinal vehicle dynamics formula $P_{\text{trac}} = \frac{1}{2}\rho\mu_d f v^3$ to capture the speed‑range trade‑off ("high speed $\rightarrow$ short range").
*   **MDP Formulation**: Formalizes the problem as a four‑tuple MDP $\{S, A, T, R\}$. The state includes SoC, remaining demands, current time, and distance matrices; the action space consists of $\{\text{next node}, \text{travel speed}, \text{charging duration}\}$.
*   **Algorithm**: Proposes **AC‑AER (Actor‑Critic with Automated Entropy Regularisation)**, utilizing an MLP network integrated with softmax action masking to filter infeasible actions and automatically adjust entropy coefficients to prevent premature convergence.
*   **Reward Function**: Comprises travel time penalties, unserved customer penalties, low‑SoC safety penalties, and late‑arrival penalties.

#### Experimental Results
*   On Solomon‑style benchmarks, AC‑AER outperformed OR‑Tools‑based heuristics (ORT‑H / ORT‑H‑NC).
*   Demonstrated that accounting for Speed‑Varying Range (SVR) and en‑route partial recharging significantly reduces total delivery time.

### 1.2 Tang et al. (2026)
**Tang, M., Yu, N., Karamouzas, I. and Ye, Z. (2026)** *'TERRAN: A transformer‑based electric vehicle routing agent for real‑time adaptive navigation'*, IEEE Transactions on Automation Science and Engineering, 23, pp. 3889–3901.

#### Main Contributions
*   **MDP Modeling**: The state tracks current time $\tau^t$, SoC, remaining load, and visitation history; actions select subsequent nodes; hard action masking eliminates time‑window, energy, and capacity violations.
*   **Future‑Feasibility Pruning (FFP)**: Extends beyond single‑step feasibility by verifying whether the vehicle can reach a charging station or depot from a candidate node, preventing energy stranding and guaranteeing 100% feasibility.
*   **Staged Reward Scheduling**: Employs a tri‑phasic training scheme: initial service rewards plus loop penalties to encourage exploration; mid‑stage low‑SoC incentives to promote charging behavior; and final pure distance minimization to alleviate sparse reward dilemmas.
*   **Network Architecture**: Utilizes Graph Attention / Transformer Encoder to encode global node context; Decoder fuses dynamic states (time, SoC, load) for autoregressive decoding. Training is conducted via PPO augmented with GAE.
*   **Benchmarking**: Rigorously compared against CPLEX (exact solutions), VNS/TS (metaheuristics), and baseline DRL methods based on the Schneider (2014) standard EVRP‑TW model.

#### Experimental Results
*   **Scalability**: Matched CPLEX optimal solutions for small‑scale instances ($n=5$); achieved $<1.5\%$ optimality gaps for medium scales ($n=15$) with approximately a 170,000× speedup; generated 100% feasible solutions for large scales ($n=100$) within 0.47s (where CPLEX predominantly failed).
*   **Ablation Studies**: Confirmed that both FFP and staged rewards substantially improve training stability and solution quality.

---

## 2. Analysis and Reproduction Strategy

Bi et al. (2024) focuses on an **extended EVRP‑TW variant** featuring speed‑varying range and en‑route partial recharging. While innovative, its reliance on MLP‑based AC‑AER and complex action spaces presents significant replication challenges.

Conversely, Tang et al. (2026) targets the **canonical EVRP‑TW problem** using a modular Transformer‑PPO architecture. Consequently, TERRAN serves as the primary reference for this reproducibility study.

The reproduction initiative commences with the **Hard Action Masking mechanism**. This submodule constitutes the bedrock of TERRAN’s feasibility guarantees and represents a formalized, systematic evolution of the `feasible()` function prototyped in Week 1.

---

## 3. Implementation: Hard Action Masking

### 3.1 Operational Principle
At each decision epoch, the mechanism interrogates the vehicle’s instantaneous state—time, SoC, and payload—to **categorically prohibit all actions precipitating constraint violations**. Rather than imposing ex‑post penalties, the mask **precludes erroneous selections by design**.

### 3.2 Core Sub‑Modules
The composite mask adheres to the formulation specified in TERRAN:

$$ M_t(i) = M_t^{\text{time}}(i) \cdot M_t^{\text{cap}}(i) \cdot M_t^{\text{energy}}(i) $$

Where:
*   **Time‑Window Mask $M_t^{\text{time}}(i)$**: Ensures arrival occurs within $[e_i, l_i]$. Early arrivals are permitted (waiting is allowed), while late arrivals are forbidden.
*   **Capacity Mask $M_t^{\text{cap}}(i)$**: Prevents overloading by validating residual capacity against nodal demand.
*   **Energy Mask $M_t^{\text{energy}}(i)$**: Guarantees sufficient SoC to traverse the distance to the subsequent node (single‑step verification).

---

## 4. Summary

This week successfully instantiated the **Hard Action Masking submodule** foundational to the TERRAN framework. The implementation strictly conforms to the constraint‑filtering protocols delineated in Tang et al. (2026), establishing a verifiable and extensible substrate for subsequent algorithmic integrations. The literature review clarified the distinction between extended EVRP‑TW variants (Bi et al.) and the standard formulation (Tang et al.), justifying the focus on hard constraint masking as the critical first step in building a robust routing agent.

---

## References
*   Bi, X., Wang, R. and Jia, Q. (2024) 'On the speed‑varying range of electric vehicles in time‑windowed routing problems with en‑route partial re‑charging', *IEEE Transactions on Consumer Electronics*, 70(1), pp. 3650–3657.
*   Tang, M., Yu, N., Karamouzas, I. and Ye, Z. (2026) 'TERRAN: A transformer‑based electric vehicle routing agent for real‑time adaptive navigation', *IEEE Transactions on Automation Science and Engineering*, 23, pp. 3889–3901.