# Week 2 Report: Deep Reinforcement Learning for Electric Vehicle Routing with Time Windows (EVRP‑TW)

## 1. Literature Review

Two recent studies were reviewed to contextualize the current research within the evolving landscape of EVRP‑TW solutions.

### Paper I: Bi et al. (2024)
**Bi, X., Wang, R. and Jia, Q. (2024)** *'On the speed‑varying range of electric vehicles in time‑windowed routing problems with en‑route partial re‑charging'*, IEEE Transactions on Consumer Electronics, 70(1), pp. 3650–3657.

#### Key Contributions
*   **Energy Consumption Model**: Utilizes a longitudinal vehicle dynamics formula $P_{\text{trac}}=\frac{1}{2}\rho\mu_d f v^3$ to model energy consumption across varying speeds, explicitly capturing the "high speed $\rightarrow$ short range" trade‑off.
*   **MDP Formulation**: Formalizes the problem as a four‑tuple MDP $\{S, A, T, R\}$. The state encapsulates SoC, residual demands, current time, and distance matrices, while the action space comprises $\{\text{next node}, \text{travel speed}, \text{charging duration}\}$.
*   **Algorithm**: Proposes **AC‑AER (Actor‑Critic with Automated Entropy Regularisation)**. This approach leverages an MLP network integrated with a softmax action masking layer to filter infeasible actions and dynamically adjusts entropy coefficients to mitigate premature convergence.
*   **Reward Design**: The reward function synthesizes travel time penalties, unserved customer penalties, low‑SoC safety penalties, and late‑arrival penalties.

#### Experimental Validation
*   On Solomon‑style benchmarks, AC‑AER outperformed OR‑Tools‑based heuristics (ORT‑H/ORT‑H‑NC).
*   Empirical results substantiate that incorporating Speed‑Varying Range (SVR) and en‑route partial recharging significantly curtails total delivery time.

### Paper II: Tang et al. (2026)
**Tang, M., Yu, N., Karamouzas, I. and Ye, Z. (2026)** *'TERRAN: A transformer‑based electric vehicle routing agent for real‑time adaptive navigation'*, IEEE Transactions on Automation Science and Engineering, 23, pp. 3889–3901.

#### Key Contributions
*   **MDP Modeling**: The state tracks current time $\tau^t$, SoC, remaining load, and visitation sequences. Actions select subsequent nodes, enforced by hard action masking to eliminate violations regarding time windows, energy levels, and capacity.
*   **Future‑Feasibility Pruning (FFP)**: Extends beyond single‑step validation by verifying whether the vehicle can reach a charging station or depot from a candidate node, fundamentally preventing energy stranding and guaranteeing 100% feasibility.
*   **Staged Reward Scheduling**: Employs a tri‑phasic training regimen: initial service rewards plus loop penalties to foster exploration; intermediate incentives for low‑SoC scenarios to encourage charging; and final pure distance minimization to alleviate sparse reward dilemmas.
*   **Network Architecture**: Utilizes a Graph Attention/Transformer Encoder to contextualize global node features, coupled with a Decoder that fuses dynamic states (time, SoC, load) for autoregressive decoding. Training is conducted via PPO augmented with GAE.
*   **Benchmarking**: Rigorously compared against CPLEX (exact solutions), VNS/TS (metaheuristics), and baseline DRL methods, utilizing the canonical Schneider (2014) EVRP‑TW model.

#### Experimental Validation
*   **Scalability**: Matched CPLEX optima for small‑scale instances ($n=5$); achieved $<1.5\%$ optimality gaps for medium scales ($n=15$) with approximately a 170,000× speedup; generated 100% feasible solutions for large scales ($n=100$) in 0.47s (where CPLEX predominantly failed).
*   **Ablation Studies**: Confirmed that both FFP and staged rewards markedly enhance training stability and solution fidelity.

---

## 2. Critical Analysis and Reproduction Strategy

Bi et al. (2024) primarily addresses an **extended EVRP‑TW variant** featuring speed‑varying range and en‑route partial recharging. While innovative, its reliance on MLP‑based AC‑AER and complex action spaces presents significant replication challenges.

Conversely, Tang et al. (2026) targets the **canonical EVRP‑TW problem** using a modular Transformer‑PPO architecture. Consequently, TERRAN serves as the principal reference for this reproducibility study.

The reproduction initiative commences with the **Hard Action Masking mechanism**. This submodule constitutes the bedrock of TERRAN’s feasibility guarantees and represents a formalized, systematic evolution of the `feasible()` function prototyped in Week 1.

### 2.1 Operational Principle
At each decision epoch, the mechanism interrogates the vehicle’s instantaneous state—time, SoC, and payload—to **prohibit categorically all actions precipitating constraint violations**. Rather than imposing ex‑post penalties, the mask **precludes erroneous selections by design**.

### 2.2 Core Sub‑Modules
The composite mask adheres to the formulation specified in TERRAN:

$$ M_t(i) = M_t^{\text{time}}(i) \cdot M_t^{\text{cap}}(i) \cdot M_t^{\text{energy}}(i) $$

Where:
*   **Time‑Window Mask $M_t^{\text{time}}(i)$**: Ensures arrival occurs within $[e_i, l_i]$, permitting early arrivals (waiting) but forbidding late arrivals.
*   **Capacity Mask $M_t^{\text{cap}}(i)$**: Prevents overloading by validating residual capacity against nodal demand.
*   **Energy Mask $M_t^{\text{energy}}(i)$**: Guarantees sufficient SoC to traverse the distance to the subsequent node (single‑step verification; FFP integration is deferred).

---

## 3. Implementation Details

The implementation prioritizes clarity and adherence to the TERRAN pseudocode logic.

### 3.1 Preliminaries: State and Environment Definition
The initial phase involved defining static parameters, nodal attributes (Depot, Customers, Charging Stations), Euclidean distance matrices, and travel‑time matrices predicated on a constant velocity assumption.

![Initialization of parameters, node definitions, and distance/time matrices](figures/fig1_mask_code.png)
*Figure 1: Configuration of environmental parameters, node metadata (coordinates, demand, time windows), and pre‑computation of distance metrics.*

### 3.2 State Representation
A dedicated `State` class was engineered to encapsulate dynamic vehicular attributes during the routing process, including the current node index, simulation clock, instantaneous SoC, residual load, and a registry of visited clients.

![Definition of the State class and masking logic](figures/fig2_state_class.png)
*Figure 2: Architectural overview of the `State` class tracking vehicle telemetry and the initialization of the feasibility assessment logic.*

### 3.3 Core Masking Modules
Three autonomous masking functions were implemented to enforce hard constraints rigorously:

![Implementation of sub-masks and composite logic](figures/fig3_energy_mask.png)
*Figure 3: Detailed implementation of the energy mask ($M_t^{\text{energy}}$) and the aggregation logic synthesizing temporal, capacitive, and energetic constraints into a unified feasibility vector.*

---

## 4. Experimental Validation and Results

### 4.1 Execution Output
Execution of the primary script (`python evrptw_mask.py`) yields step‑wise feasibility masks. As illustrated in Figure 4, the mechanism accurately delineates permissible nodes at each juncture. At the inaugural step ($t=0$), all nodes (Depot, C1, C2, CS) are deemed feasible (Mask: `[0 1 1 1]`), thereby enabling the agent to explore legitimate trajectories.

![Terminal output of the masking demonstration](figures/fig4_terminal_output.png)
*Figure 4: Terminal interface displaying the dynamic generation of feasibility masks across discrete decision epochs and the enumeration of admissible actions.*

### 4.2 Unit Test Verification
A bespoke test suite (`test_mask.py`) was devised to validate the robustness of the masking logic under induced boundary conditions:

*   **Time‑Window Violation**: Confirmed interdiction of actions resulting in tardy arrivals.
*   **Capacity Violation**: Validated rejection of actions exceeding vehicular payload thresholds.
*   **Energy Insufficiency**: Ensured blocking of actions precipitating battery depletion.
*   **Charging Accessibility**: Affirmed perpetual accessibility of Charging Stations under nominal constraints.

![Results of the unit test suite](figures/fig5_test_results.png)
*Figure 5: Successful execution of the diagnostic test suite, evidencing the algorithmic integrity of the masking protocol.*

All test vectors terminated successfully (`All tests passed!`), furnishing empirical validation of the algorithm’s correctness and reliability.

---

## 5. Conclusion and Future Work

### 5.1 Conclusion
This week successfully instantiated the **Hard Action Masking submodule** foundational to the TERRAN framework. The implementation strictly conforms to the constraint‑filtering protocols delineated in Tang et al. (2026), establishing a verifiable and extensible substrate for subsequent algorithmic integrations.

### 5.2 Future Work
1.  **Integration of FFP**: Augment the extant single‑step energy audit with multi‑step Future‑Feasibility Pruning to ensure perpetual vehicular solvency regarding energy reservoirs.
2.  **Policy Synthesis**: Formulate a rudimentary greedy rollout policy contingent upon the generated masks to simulate end‑to‑end routing behaviors.
3.  **Benchmark Scaling**: Transition from pedagogical toy instances to standardized Solomon‑style EVRP‑TW benchmarks to facilitate rigorous quantitative evaluation.

---

## References
*   Bi, X., Wang, R. and Jia, Q. (2024) 'On the speed‑varying range of electric vehicles in time‑windowed routing problems with en‑route partial re‑charging', *IEEE Transactions on Consumer Electronics*, 70(1), pp. 3650–3657.
*   Tang, M., Yu, N., Karamouzas, I. and Ye, Z. (2026) 'TERRAN: A transformer‑based electric vehicle routing agent for real‑time adaptive navigation', *IEEE Transactions on Automation Science and Engineering*, 23, pp. 3889–3901.