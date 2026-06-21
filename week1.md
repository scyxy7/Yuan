# Week 1 Report  
## Deep Reinforcement Learning for Electric Vehicle Routing with Time Windows (EVRP‑TW)

---

## 1. Introduction
With the rapid electrification of road transport, Electric Vehicles (EVs) are increasingly recognized as a key solution to reduce greenhouse gas emissions in urban logistics. However, large-scale adoption introduces new operational challenges, especially when EVs must respect time windows (TW) and limited battery capacity.

The Electric Vehicle Routing Problem with Time Windows (EVRP‑TW) extends the classical Vehicle Routing Problem (VRP) by incorporating recharging stations, energy consumption constraints, and strict delivery schedules. Solving EVRP‑TW at scale is NP‑hard, making exact optimization impractical for real‑world instances.

This project investigates Deep Reinforcement Learning (Deep RL) as a scalable alternative to traditional Operations Research (OR) methods for learning constructive routing policies.

---

## 2. Problem Definition (Based on Energies 2022)

The EVRP‑TW is formally defined over a directed graph G = (V, E), where:

V = {0} ∪ C ∪ S  
- 0: depot  
- C: customer nodes with time windows [e_i, l_i]  
- S: charging stations (optional, repeatable visits)

Each EV has:
- Battery capacity Q  
- Initial State of Charge (SOC) q₀  
- Energy consumption eᵢⱼ per arc (i, j)

### Constraints (Casella et al., 2022, Sec. 4)
1. Each customer is visited exactly once  
2. Vehicle battery never drops below zero  
3. Arrival time must satisfy time windows  
4. Recharging is allowed at CS nodes (full or partial)

### Objective
Minimize total travel cost:

min ∑ₖ ∑₍ᵢ,ⱼ₎∈E cᵢⱼ xᵢⱼₖ

---

## 3. Literature Review

### 3.1 Classical EVRP‑TW Studies
Casella et al. (2022) classify EVRP‑TW as one of the most challenging VRP variants due to the coupling of energy and temporal constraints.

Reference — Contribution  
Schneider et al. (2014) — First formal EVRP‑TW with recharging stations  
Desaulniers et al. (2016) — Exact branch‑price‑and‑cut for EVRP‑TW  
Keskin & Çatay (2016) — ALNS with partial recharging  
Hiermann et al. (2016) — Mixed fleet EVRP‑TW  
Pelletier et al. (2019) — Energy consumption uncertainty  

These works confirm that exact methods do not scale, motivating learning‑based approaches.

---

### 3.2 Limitations of Traditional Methods
As noted in the review:
- Exact methods fail beyond small instances  
- Metaheuristics require hand‑crafted operators  
- Limited generalization across instance sizes  

This creates space for Deep RL–based Neural Combinatorial Optimization (NCO).

---

## 4. Proposed Research Direction

This project will:
1. Model EVRP‑TW as a Markov Decision Process (MDP)  
2. Use a constructive policy (encoder–decoder with attention)  
3. Train with policy gradient methods (REINFORCE / PPO)  
4. Compare against classical EVRP‑TW benchmarks

### Preliminary MDP Formulation

Element — Description  
State sₜ — Current node, remaining SOC, current time, visited set  
Action aₜ — Next node (customer or charging station)  
Reward rₜ — −cᵢⱼ for travel cost  
Termination — All customers served or infeasible  

Action masking will enforce time window feasibility and SOC feasibility.

---

## 5. Environment Setup and Preliminary Test

The development environment was configured as follows:

conda create -n evrptw python=3.9  
conda activate evrptw  
pip install torch numpy matplotlib tqdm

A minimal feasibility test was implemented to validate energy and time window constraints, inspired by the EVRP‑TW formulation in Casella et al. (2022):

def feasible(next_node, soc, time, e, l, energy_cost):
    if soc - energy_cost < 0:
        return False
    if not (e <= time <= l):
        return False
    return True

This test demonstrates that constraint checking can be implemented independently of the learning algorithm.

---

## 6. Weekly Progress and Plan

### Completed
- Reviewed Energies (2022) survey on EVRP and EV integration  
- Clarified EVRP‑TW problem structure  
- Installed Python and PyTorch environment  
- Implemented basic feasibility check

### Week 2 Plan
- Reproduce Attention Model (Kool et al., 2019) on CVRP  
- Extend to VRPTW  
- Begin EVRP‑TW state representation design

---

## 7. References
Casella, V.; Fernandez Valderrama, D.; Ferro, G.; Minciardi, R.; Paolucci, M.; Parodi, L.; Robba, M. Towards the Integration of Sustainable Transportation and Smart Grids: A Review on Electric Vehicles’ Management. Energies, 2022, 15, 4020.

Schneider, M.; Stenger, A.; Goeke, D. The electric vehicle‑routing problem with time windows and recharging stations. Transportation Science, 2014.

Kool, W.; van Hoof, H.; Welling, M. Attention, Learn to Solve Routing Problems. ICLR, 2019.