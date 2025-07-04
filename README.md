<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>Universiteti i Prishtines</p>
      <p>Fakulteti i Inxhinierise Elektrike dhe Kompjuterike</p>
      <p>Programi Master</p>
      <p>Profesori: Prof. Dr. Kadri Sulejman</p>
      <p>Lenda: Algoritmet e inspiruara nga natyra</p>
    </td>
 </tr>
</table>

## Project Overview
This repository implements advanced metaheuristic optimization for the "Book Scanning" problem, using the **Great Deluge Algorithm (GDA)** and several enhancements. The project is designed for the course "Algoritmet e Inspiruara nga Natyra" (Nature-Inspired Algorithms) and demonstrates adaptive, configurable, and parallelized approaches to combinatorial optimization.

## How the Great Deluge Algorithm Works
The Great Deluge Algorithm is a metaheuristic inspired by the idea of a receding water level (boundary). The algorithm starts with an initial solution and a high boundary ("water level"). At each step:
- A new solution (neighbor) is generated by tweaking the current one.
- If the new solution's score is above the current boundary, it is accepted.
- The boundary drops over time, making the acceptance criterion stricter.
- The process continues until time or iteration limits are reached.

**Water Analogy:**
- Imagine a landscape of hills (solutions) and a water level that drops over time.
- You can only move to hills above the water (solutions above the boundary).
- As the water drops, only the highest hills (best solutions) remain accessible.

## Configuration (JSON)
The algorithm is highly configurable via JSON files (e.g., `enhanced_gda_config.json`). The config controls:
- **Parameter sets:** Time limits, boundary buffer, iteration counts, etc.
- **Boundary functions:** How the acceptance threshold (boundary) drops (e.g., linear, exponential, adaptive).
- **Neighbor selection:** Probabilities for each tweak method (e.g., swap libraries, insert library).
- **Adaptive strategies:** Multi-phase search, dynamic parameter adjustment, and fast/parallel execution.

## Key Methods
- **great_deluge_algorithm_configurable:** Main GDA loop, configurable via JSON.
- **run_parallel_gda_from_json:** Loads config, runs GDA in parallel for each parameter set.
- **generate_initial_solution_grasp:** Creates an initial solution using GRASP.
- **Tweak methods:**
  - `tweak_solution_swap_signed`: Swap two signed libraries.
  - `tweak_solution_swap_same_books`: Swap books between libraries.
  - `tweak_solution_insert_library`: Insert an unsigned library.
  - `tweak_solution_swap_last_book`: Swap last book between libraries.
  - `tweak_solution_swap_signed_with_unsigned`: Swap signed/unsigned libraries.
  - `tweak_solution_swap_neighbor_libraries`: Swap adjacent libraries.
  - `hill_climbing_combined`: Local search using multiple tweaks.
  - `perturb_solution`: Applies random tweaks for restarts.
- **Boundary functions:**
  - `linear`, `multiplicative`, `stepwise`, `logarithmic`, `quadratic`, `hybrid_adaptive`, `sigmoid_decay`, `oscillating_decay`.

## How to Run
1. **Run the algorithm:**
   ```bash
   python gda_tuning.py
   ```
   - Selects mode (fast/enhanced), loads config, and processes all input files.
2. **Validate solutions:**
   - Use the GUI or CLI in `validator/validator.py` to check solution validity and score.

## Directory Structure
- `app.py`, `gda_tuning.py`: Main scripts for running experiments.
- `models/`: Core logic, including the Great Deluge algorithm and solution tweaks.
- `generator/`: Tools for generating problem instances.
- `input/`: Contains problem instance files.
- `validator/`: Solution validation tools (GUI and CLI).
- `*.json`: Configuration and result files.

---

For more details, see the code comments and configuration files. If you have questions about specific methods or want to extend the algorithm, see the `models/solver.py` and `models/GreatDeluge/great_deluge.py` files.
