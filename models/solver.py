import random
from collections import defaultdict
import threading
import time
from models.GreatDeluge.great_deluge import ParallelGDARunner
from models.library import Library
import os
from models.solution import Solution
import copy
import random
import math
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from typing import Tuple
from models.instance_data import InstanceData
import gc  # Add garbage collection support

class Solver:
    def __init__(self):
        pass
    
    def great_deluge_algorithm_configurable(self, data, **kwargs):
        """
        Enhanced Configurable Great Deluge Algorithm with adaptive capabilities
        """
        import random
        import math
        
        # Extract parameters with defaults
        max_time = kwargs.get('max_time', 300)
        max_iterations = kwargs.get('max_iterations', 1000)
        initial_boundary_buffer = kwargs.get('initial_boundary_buffer', 1.25)
        delta_B_factor = kwargs.get('delta_B_factor', 0.3)
        alpha = kwargs.get('alpha', 0.92)
        beta = kwargs.get('beta', 1.05)
        memory_window_size = kwargs.get('memory_window_size', 75)
        improvement_threshold_factor = kwargs.get('improvement_threshold_factor', 0.002)
        decay_function = kwargs.get('decay_function', 'linear')
        restart_stagnation_count = kwargs.get('restart_stagnation_count', 50)
        neighbor_selection = kwargs.get('neighbor_selection', {
            'swap_signed': 0.33,
            'swap_same_books': 0.33,
            'insert_library': 0.34
        })
        
        # Problem-size adaptive adjustments
        num_books = len(data.scores) if hasattr(data, 'scores') else 0
        num_libraries = len(data.libs) if hasattr(data, 'libs') else 0
        
        # Apply problem-size adaptive scaling with fast mode acceleration
        fast_mode_enabled = kwargs.get('fast_mode_enabled', False)
        time_reduction = kwargs.get('time_reduction_factor', 1.0) if fast_mode_enabled else 1.0
        
        if num_books < 1000 and num_libraries < 100:  # Small instance
            max_time = int(max_time * 0.7 * time_reduction)
            initial_boundary_buffer *= 0.9
            restart_stagnation_count = int(restart_stagnation_count * 0.8)
            max_iterations = int(max_iterations * 1.2)
        elif num_books >= 10000 or num_libraries >= 1000:  # Large instance
            max_time = int(max_time * 1.5 * time_reduction)
            initial_boundary_buffer *= 1.2
            restart_stagnation_count = int(restart_stagnation_count * 1.3)
            max_iterations = int(max_iterations * 0.8)
        else:  # Medium instance
            max_time = int(max_time * time_reduction)
        
        if fast_mode_enabled:
            print(f"⚡ Fast mode: Reduced execution time by {int((1-time_reduction)*100)}%")
        
        # Validate input type
        if not hasattr(data, 'libs'):
            raise TypeError("First argument must be problem Data instance")

        # Initialize with GRASP-generated solution
        current_solution = self.generate_initial_solution_grasp(data, p=0.1, max_time=30)
        current_score = current_solution.fitness_score
        best_solution = copy.deepcopy(current_solution)
        best_score = current_score
        
        # Configurable boundary initialization
        B = current_score * initial_boundary_buffer
        original_B = B
        
        # Configurable decay calculation
        delta_B = (current_score * delta_B_factor) / max_iterations
        original_delta_B = delta_B
            
        # Memory structures for stagnation detection
        memory_window = deque(maxlen=memory_window_size)
        improvement_threshold = current_score * improvement_threshold_factor
        
        # Multi-phase strategy variables
        total_phases = 3
        phase_durations = [0.4, 0.35, 0.25]  # exploration, intensification, diversification
        current_phase = 0
        phase_start_time = 0
        phase_best_score = current_score
        
        # Dynamic parameter adjustment variables
        performance_window = deque(maxlen=50)
        last_adjustment_iteration = 0
        adjustment_frequency = 100
        
        start_time = time.time()
        iterations = 0
        stagnation_count = 0
        
        while (time.time() - start_time) < max_time and iterations < max_iterations:
            try:
                # Multi-phase strategy: adjust parameters based on current phase
                elapsed_ratio = (time.time() - start_time) / max_time
                new_phase = 0
                cumulative_ratio = 0
                for i, ratio in enumerate(phase_durations):
                    cumulative_ratio += ratio
                    if elapsed_ratio <= cumulative_ratio:
                        new_phase = i
                        break
                else:
                    new_phase = len(phase_durations) - 1
                
                # Phase transition logic
                if new_phase != current_phase:
                    current_phase = new_phase
                    phase_start_time = time.time()
                    phase_best_score = current_score
                    
                    # Adjust parameters for new phase
                    if current_phase == 0:  # Exploration
                        neighbor_selection = kwargs.get('neighbor_selection', {
                            'swap_signed': 0.2,
                            'swap_same_books': 0.3,
                            'insert_library': 0.4,
                            'swap_signed_with_unsigned': 0.1
                        })
                        alpha *= 0.95  # More aggressive exploration
                        beta *= 1.05
                    elif current_phase == 1:  # Intensification
                        neighbor_selection = {
                            'swap_signed': 0.4,
                            'swap_same_books': 0.4,
                            'insert_library': 0.15,
                            'swap_signed_with_unsigned': 0.05
                        }
                        alpha *= 1.02  # More conservative
                        beta *= 0.98
                    else:  # Diversification
                        neighbor_selection = {
                            'swap_signed': 0.25,
                            'swap_same_books': 0.25,
                            'insert_library': 0.3,
                            'swap_signed_with_unsigned': 0.2
                        }
                        alpha = (alpha + 0.9) / 2  # Reset to balanced
                        beta = (beta + 1.05) / 2
                
                # Dynamic parameter adjustment
                if iterations - last_adjustment_iteration >= adjustment_frequency:
                    if len(performance_window) >= 10:
                        recent_improvement = max(performance_window) - min(performance_window)
                        if recent_improvement < improvement_threshold:
                            # Adjust parameters for better exploration
                            initial_boundary_buffer *= 1.1
                            delta_B_factor *= 1.05
                            restart_stagnation_count = max(10, int(restart_stagnation_count * 0.9))
                        last_adjustment_iteration = iterations
                
                # Select neighbor operator based on probabilities
                operator_choice = random.random()
                cumulative_prob = 0.0
                
                for operator, prob in neighbor_selection.items():
                    cumulative_prob += prob
                    if operator_choice <= cumulative_prob:
                        if operator == 'swap_signed':
                            neighbor = self.tweak_solution_swap_signed(current_solution, data)
                        elif operator == 'swap_same_books':
                            neighbor = self.tweak_solution_swap_same_books(current_solution, data)
                        elif operator == 'insert_library':
                            neighbor = self.tweak_solution_insert_library(current_solution, data)
                        elif operator == 'swap_last_book':
                            neighbor = self.tweak_solution_swap_last_book(current_solution, data)
                        elif operator == 'swap_signed_with_unsigned':
                            neighbor = self.tweak_solution_swap_signed_with_unsigned(current_solution, data)
                        elif operator == 'swap_neighbor_libraries':
                            neighbor = self.tweak_solution_swap_neighbor_libraries(current_solution, data)
                        else:
                            neighbor = self.hill_climbing_combined(data, iterations=15)[1]
                        break
                else:
                    # Default fallback
                    neighbor = self.hill_climbing_combined(data, iterations=15)[1]
                
                # Solution validation
                if not isinstance(neighbor, Solution):
                    raise RuntimeError("Neighbor generation failed - invalid solution type")

                neighbor_score = neighbor.fitness_score
                performance_window.append(neighbor_score)

                # Enhanced acceptance criteria
                if neighbor_score >= current_score or neighbor_score >= B:
                    current_solution = neighbor
                    current_score = neighbor_score
                    stagnation_count = 0
                    
                    # Update best solution with elite selection
                    if current_score > best_score:
                        best_solution = copy.deepcopy(current_solution)
                        best_score = current_score
                        delta_B *= alpha  # Accelerate decay
                        B = best_score * initial_boundary_buffer  # Reset boundary relative to best
                    else:
                        delta_B *= beta   # Encourage exploration
                else:
                    stagnation_count += 1

                # Enhanced boundary adjustment with new decay functions
                if decay_function == 'linear':
                    # B = B₀ - r×i (constant rate decrease)
                    B = max(B - delta_B, 0)
                elif decay_function == 'multiplicative':
                    # B = B₀ × α^i (exponential decay)
                    B = max(B * alpha, 0)
                elif decay_function == 'stepwise':
                    # B = B₀ - (i÷s)×d (step-wise decrease)
                    step_size = kwargs.get('step_size', 10)
                    d_factor = kwargs.get('d_factor', 50)
                    B = max(B - (iterations // step_size) * d_factor, 0)
                elif decay_function == 'logarithmic':
                    # B = B₀ - r×log(i+1) (logarithmic decay)
                    B = max(B - delta_B * math.log(iterations + 1), 0)
                elif decay_function == 'quadratic':
                    # B = B₀ - r×i² (quadratic decay)
                    B = max(B - delta_B * 0.01 * (iterations ** 2), 0)
                elif decay_function == 'hybrid_adaptive':
                    # Adaptive switching between strategies based on performance
                    if stagnation_count < 20:
                        B = max(B - delta_B, 0)  # Linear during good progress
                    elif stagnation_count < 40:
                        B = max(B * alpha, 0)    # Exponential during stagnation
                    else:
                        B = max(B - delta_B * math.log(iterations + 1), 0)  # Logarithmic during deep stagnation
                elif decay_function == 'sigmoid_decay':
                    # S-curve decay: B = B₀ / (1 + e^((i-mid)/scale))
                    mid_point = max_iterations / 2
                    scale = max_iterations / 6
                    sigmoid_factor = 1 / (1 + math.exp((iterations - mid_point) / scale))
                    B = max(original_B * sigmoid_factor, 0)
                elif decay_function == 'oscillating_decay':
                    # Oscillating boundary with overall decay trend
                    base_decay = original_B * (1 - iterations / max_iterations)
                    oscillation = 0.1 * original_B * math.sin(iterations * 0.1)
                    B = max(base_decay + oscillation, 0)
                else:
                    # Default to linear
                    B = max(B - delta_B, 0)
                
                # Stagnation detection and restart
                memory_window.append(current_score)
                if stagnation_count >= restart_stagnation_count:
                    # Restart from best solution with perturbation
                    current_solution = self.perturb_solution(best_solution, data)
                    current_score = current_solution.fitness_score
                    B = current_score * initial_boundary_buffer * 1.2
                    delta_B *= 0.8
                    stagnation_count = 0
                        
                iterations += 1

            except Exception as e:
                print(f"Iteration {iterations} failed: {str(e)}")
                break

        return best_score, best_solution

    def run_parallel_gda_from_json(self, config_file, data, output_file='gda_results.json', input_file=None, mode='enhanced', folder_type='old-instances', max_threads_override=None):
        """
        Enhanced method: Load JSON config with adaptive capabilities and run optimizations in parallel
        
        Args:
            config_file: Path to enhanced JSON configuration file
            data: Problem instance data
            output_file: Path to save results
            input_file: Path to input file (used for solution naming)
            max_threads_override: Override maximum thread count for large instances
            
        Returns:
            Enhanced results with adaptive strategy performance
        """
        import json
        import time
        import threading
        import random
        from concurrent.futures import ThreadPoolExecutor
        
        # Force garbage collection at start for large instances
        if max_threads_override:
            print(f"   🧹 Pre-processing cleanup for large instance...")
            gc.collect()
        
        # Load configuration
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found")
            return None
        
        # Extract enhanced configuration sections
        param_configs = config.get('gda_parameters', [])
        boundary_functions = config.get('boundary_functions', ['linear', 'multiplicative', 'stepwise', 'logarithmic', 'quadratic'])
        enhanced_boundaries = config.get('enhanced_boundary_functions', boundary_functions)
        adaptive_ranges = config.get('adaptive_parameter_ranges', {})
        problem_size_adaptive = config.get('problem_size_adaptive', {})
        multi_phase_strategy = config.get('multi_phase_strategy', {})
        dynamic_adjustment = config.get('dynamic_parameter_adjustment', {})
        execution_method = config.get('execution_method', 'parallel_threads')
        fast_mode = config.get('fast_execution_mode', {})
        
        # Problem size analysis
        num_books = len(data.scores) if hasattr(data, 'scores') else 0
        num_libraries = len(data.libs) if hasattr(data, 'libs') else 0
        
        print(f"🔧 Enhanced GDA Configuration Analysis")
        print(f"📖 Problem size: {num_books} books, {num_libraries} libraries")
        
        # Determine problem size category and apply adaptive adjustments
        problem_category = "medium_instance"  # default
        if num_books < 1000 and num_libraries < 100:
            problem_category = "small_instance"
        elif num_books >= 10000 or num_libraries >= 1000:
            problem_category = "large_instance"
        
        print(f"🎯 Problem category: {problem_category}")
        
        # Apply problem-size adaptive adjustments to parameter configs
        size_config = problem_size_adaptive.get(problem_category, {})
        if size_config:
            time_multiplier = size_config.get('time_multiplier', 1.0)
            boundary_adjustment = size_config.get('boundary_buffer_adjustment', 1.0)
            stagnation_factor = size_config.get('stagnation_factor', 1.0)
            iteration_boost = size_config.get('iteration_boost', 1.0)
            
            # Apply adjustments to all parameter configurations
            for param_config in param_configs:
                param_config['max_time'] = int(param_config.get('max_time', 60) * time_multiplier)
                param_config['initial_boundary_buffer'] = param_config.get('initial_boundary_buffer', 1.25) * boundary_adjustment
                param_config['restart_stagnation_count'] = int(param_config.get('restart_stagnation_count', 50) * stagnation_factor)
                param_config['max_iterations'] = int(param_config.get('max_iterations', 1000) * iteration_boost)
        
        # Generate adaptive parameter combinations from ranges (optimized for fast mode)
        adaptive_configs = []
        if adaptive_ranges:
            # Reduce variations in fast mode
            variations_per_phase = 1 if fast_mode.get('enabled') else 3
            for phase_name, phase_config in adaptive_ranges.items():
                # Generate combinations from parameter arrays
                for i in range(variations_per_phase):
                    adaptive_config = {}
                    for param, value in phase_config.items():
                        if isinstance(value, list) and param != 'neighbor_selection':
                            # In fast mode, prefer middle values; otherwise random
                            if fast_mode.get('enabled') and len(value) > 1:
                                adaptive_config[param] = value[len(value)//2]  # Pick middle value
                            else:
                                adaptive_config[param] = random.choice(value)
                        else:
                            adaptive_config[param] = value
                    adaptive_config['name'] = f"{phase_name}_var_{i+1}"
                    adaptive_configs.append(adaptive_config)
        
        # Combine original configs with adaptive configs
        all_configs = param_configs + adaptive_configs
        
        # If old format (single config), convert to array
        if isinstance(all_configs, dict):
            all_configs = [all_configs]
        
        # Use enhanced boundary functions if available
        all_boundary_functions = enhanced_boundaries if enhanced_boundaries else boundary_functions
        
        # Create all combinations of parameters × boundary functions (with fast mode limiting)
        test_combinations = []
        for param_config in all_configs:
            for boundary_func in all_boundary_functions:
                test_combinations.append((param_config, boundary_func))
        
        # Apply fast mode limitations (only if fast_execution_mode is enabled in config)
        if fast_mode.get('enabled'):
            max_combinations = fast_mode.get('max_combinations', 40)
            if len(test_combinations) > max_combinations:
                # Keep best strategies if specified, otherwise take first N
                if fast_mode.get('prefer_best_strategies'):
                    # Prioritize combinations with proven effective strategies
                    priority_boundaries = ['hybrid_adaptive', 'oscillating_decay', 'linear', 'multiplicative']
                    priority_configs = ['fast_hybrid', 'fast_exploration', 'fast_balanced']
                    
                    prioritized = []
                    remaining = []
                    
                    for combo in test_combinations:
                        config_name = combo[0].get('name', '')
                        boundary = combo[1]
                        
                        if any(pc in config_name for pc in priority_configs) or boundary in priority_boundaries:
                            prioritized.append(combo)
                        else:
                            remaining.append(combo)
                    
                    # Take prioritized first, then fill with remaining
                    test_combinations = (prioritized + remaining)[:max_combinations]
                else:
                    test_combinations = test_combinations[:max_combinations]
                
                print(f"⚡ Fast mode: Limited to {len(test_combinations)} combinations (from {len(all_configs) * len(all_boundary_functions)})")
        elif mode == 'fast':
            # If mode is 'fast' but config doesn't have fast_execution_mode, apply basic limitations
            max_combinations = 28  # Fast mode default from original implementation
            if len(test_combinations) > max_combinations:
                test_combinations = test_combinations[:max_combinations]
                print(f"⚡ Fast mode (basic): Limited to {len(test_combinations)} combinations")
        
        print(f"📊 {'Enhanced' if mode == 'enhanced' else 'Fast'} Configuration Summary:")
        print(f"   • Base parameter configurations: {len(param_configs)}")
        print(f"   • Adaptive range configurations: {len(adaptive_configs)}")
        print(f"   • Total parameter configs: {len(all_configs)}")
        print(f"   • Boundary functions: {len(all_boundary_functions)}")
        print(f"   • Total combinations: {len(test_combinations)}")
        print(f"   • Multi-phase strategy: {'enabled' if multi_phase_strategy.get('enabled') else 'disabled'}")
        print(f"   • Dynamic adjustment: {'enabled' if dynamic_adjustment.get('enabled') else 'disabled'}")
        
        # Display fast mode optimizations if applicable
        if fast_mode.get('enabled') or mode == 'fast':
            time_reduction = int((1-fast_mode.get('time_reduction_factor', 0.3))*100) if fast_mode.get('enabled') else 30
            print(f"   • Fast mode optimizations: enabled ({time_reduction}% time reduction)")
        
        print(f"🚀 Executing in parallel...")
        
        # Thread-safe results storage
        results = []
        results_lock = threading.Lock()
        
        def test_parameter_boundary_combination(combination_data):
            """Test a specific parameter config + boundary function combination with enhanced features"""
            param_config, boundary_func = combination_data
            config_name = param_config.get('name', 'unnamed')
            
            # Prepare test parameters with enhanced features
            test_params = param_config.copy()
            test_params['decay_function'] = boundary_func
            
            # Add multi-phase strategy parameters if enabled
            if multi_phase_strategy.get('enabled'):
                test_params['multi_phase_enabled'] = True
                test_params['phase_duration_ratio'] = multi_phase_strategy.get('phase_duration_ratio', [0.4, 0.35, 0.25])
                test_params['adaptive_phase_switching'] = multi_phase_strategy.get('adaptive_phase_switching', True)
            
            # Add dynamic adjustment parameters if enabled
            if dynamic_adjustment.get('enabled'):
                test_params['dynamic_adjustment_enabled'] = True
                test_params['adjustment_frequency'] = dynamic_adjustment.get('adjustment_frequency', 100)
                test_params['performance_window'] = dynamic_adjustment.get('performance_window', 50)
                test_params['improvement_threshold'] = dynamic_adjustment.get('improvement_threshold', 0.01)
            
            # Add fast mode parameters if enabled
            if fast_mode.get('enabled'):
                test_params['fast_mode_enabled'] = True
                test_params['time_reduction_factor'] = fast_mode.get('time_reduction_factor', 0.3)
            elif mode == 'fast':
                # If mode is 'fast' but config doesn't specify, apply basic time reduction
                test_params['fast_mode_enabled'] = True
                test_params['time_reduction_factor'] = 0.7  # 30% time reduction
            
            # Remove the name field as it's not needed for algorithm
            test_params.pop('name', None)
            
            try:
                start_time = time.time()
                score, solution = self.great_deluge_algorithm_configurable(data, **test_params)
                end_time = time.time()
                
                # Store result without saving individual solution files
                # (Only best solution will be saved later)
                result = {
                    'config_name': config_name,
                    'boundary_function': boundary_func,
                    'score': score,
                    'execution_time': round(end_time - start_time, 2),
                    'solution': solution,  # Keep solution object for best selection
                    'problem_category': problem_category,
                    'enhanced_features': {
                        'multi_phase': multi_phase_strategy.get('enabled', False),
                        'dynamic_adjustment': dynamic_adjustment.get('enabled', False),
                        'problem_size_adaptive': bool(size_config)
                    }
                }
                
                with results_lock:
                    results.append(result)
                
                print(f"✓ {config_name} + {boundary_func}: {score} (time: {result['execution_time']}s)")
                
                # Memory cleanup for large instances
                if max_threads_override and len(results) % 4 == 0:
                    gc.collect()
                
            except Exception as e:
                error_result = {
                    'config_name': config_name,
                    'boundary_function': boundary_func,
                    'score': 0,
                    'execution_time': 0,
                    'error': str(e),
                    'solution': None
                }
                
                with results_lock:
                    results.append(error_result)
                
                print(f"✗ {config_name} + {boundary_func}: FAILED - {str(e)}")
        
        # Execute all combinations in parallel with optimal thread count
        start_total_time = time.time()
        
        # Calculate optimal thread count with enhanced scaling for faster execution
        # Apply override if specified for large instances
        if max_threads_override:
            optimal_threads = max_threads_override
            print(f"🧵 Using {optimal_threads} parallel threads for {len(test_combinations)} combinations (override for large instance)")
        else:
            optimal_threads = min(
                len(test_combinations),  # Don't exceed number of tasks
                max(16, len(test_combinations) // 4),  # More aggressive scaling (min 16, 1 thread per 4 tasks)
                64  # Increased maximum limit for high-performance systems
            )
            mode_text = "fast mode" if fast_mode.get('enabled') else mode
            print(f"🧵 Using {optimal_threads} parallel threads for {len(test_combinations)} combinations ({mode_text})")
        
        with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
            executor.map(test_parameter_boundary_combination, test_combinations)
        
        end_total_time = time.time()
        total_execution_time = round(end_total_time - start_total_time, 2)
        
        # Force cleanup after parallel execution for large instances
        if max_threads_override:
            print(f"   🧹 Post-processing cleanup...")
            gc.collect()
        
        # Sort results by score (descending)
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Enhanced analysis and summary
        if results:
            best_result = results[0]
            avg_score = sum(r.get('score', 0) for r in results) / len(results)
            successful_runs = [r for r in results if r.get('score', 0) > 0]
            
            # Analyze performance by features
            multi_phase_results = [r for r in results if r.get('enhanced_features', {}).get('multi_phase')]
            adaptive_results = [r for r in results if r.get('enhanced_features', {}).get('dynamic_adjustment')]
            
            print(f"\n📈 Enhanced Results Summary:")
            print(f"   • Total execution time: {total_execution_time}s")
            print(f"   • Successful runs: {len(successful_runs)}/{len(results)}")
            print(f"   • Best score: {best_result.get('score', 0)} ({best_result.get('config_name')} + {best_result.get('boundary_function')})")
            print(f"   • Average score: {avg_score:.2f}")
            
            if multi_phase_results:
                multi_phase_avg = sum(r.get('score', 0) for r in multi_phase_results) / len(multi_phase_results)
                print(f"   • Multi-phase average: {multi_phase_avg:.2f}")
            
            if adaptive_results:
                adaptive_avg = sum(r.get('score', 0) for r in adaptive_results) / len(adaptive_results)
                print(f"   • Dynamic adjustment average: {adaptive_avg:.2f}")
            
            print(f"\n🏆 Top 5 Enhanced Combinations:")
            for i, result in enumerate(results[:5], 1):
                features = result.get('enhanced_features', {})
                feature_flags = []
                if features.get('multi_phase'): feature_flags.append('MP')
                if features.get('dynamic_adjustment'): feature_flags.append('DA')
                if features.get('problem_size_adaptive'): feature_flags.append('PSA')
                feature_str = f" [{'/'.join(feature_flags)}]" if feature_flags else ""
                
                print(f"   {i}. {result.get('config_name')} + {result.get('boundary_function')}: {result.get('score')}{feature_str}")
        
        # Save only the best solution to output/jsontorun with input filename
        best_solution_file = None
        if results and input_file and results[0].get('solution'):
            try:
                import os
                # Extract filename without extension from input file
                input_filename = os.path.basename(input_file).replace('.txt', '')
                best_solution_file = f"output/jsontorun/{folder_type}/{mode}/{input_filename}.txt"
                
                # Ensure output directory exists
                os.makedirs(f"output/jsontorun/{folder_type}/{mode}", exist_ok=True)
                
                # Export only the best solution
                best_solution = results[0]['solution']
                best_solution.export(best_solution_file)
                
                print(f"💾 Best solution saved to: {best_solution_file}")
                print(f"🏆 Best combination: {results[0].get('config_name')} + {results[0].get('boundary_function')} (Score: {results[0].get('score')})")
                
            except Exception as e:
                print(f"⚠️  Failed to save best solution: {str(e)}")
                best_solution_file = None
        
        # Clean results for JSON serialization (remove solution objects)
        clean_results = []
        for result in results:
            clean_result = result.copy()
            if 'solution' in clean_result:
                del clean_result['solution']  # Remove non-serializable solution object
            # Add solution file info only for the best result
            clean_result['solution_file'] = best_solution_file if result == results[0] and best_solution_file else None
            clean_results.append(clean_result)
        
        # Enhanced output with metadata
        enhanced_output = {
            'metadata': {
                'config_file': config_file,
                'input_file': input_file,
                'execution_time': total_execution_time,
                'problem_size': {
                    'books': num_books,
                    'libraries': num_libraries,
                    'category': problem_category
                },
                'enhanced_features': {
                    'adaptive_ranges': len(adaptive_configs),
                    'multi_phase_strategy': multi_phase_strategy.get('enabled', False),
                    'dynamic_adjustment': dynamic_adjustment.get('enabled', False),
                    'problem_size_adaptive': bool(size_config)
                },
                'total_combinations': len(test_combinations),
                'successful_runs': len([r for r in results if r.get('score', 0) > 0]),
                'threads_used': optimal_threads,
                'memory_optimization': bool(max_threads_override)
            },
            'results': clean_results,
            'summary': {
                'best_score': results[0].get('score', 0) if results else 0,
                'best_combination': f"{results[0].get('config_name', 'N/A')} + {results[0].get('boundary_function', 'N/A')}" if results else "N/A",
                'average_score': sum(r.get('score', 0) for r in results) / len(results) if results else 0,
                'top_3_combinations': [
                    {
                        'rank': i+1,
                        'config': result.get('config_name'),
                        'boundary': result.get('boundary_function'),
                        'score': result.get('score'),
                        'enhanced_features': result.get('enhanced_features', {})
                    }
                    for i, result in enumerate(results[:3])
                ]
            }
        }
        
        # Add best solution file to metadata
        enhanced_output['metadata']['best_solution_file'] = best_solution_file
        
        # Save enhanced results (cleaned for JSON serialization)
        try:
            with open(output_file, 'w') as f:
                json.dump(enhanced_output, f, indent=2)
            print(f"📁 Enhanced results saved to {output_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results to {output_file}: {str(e)}")
        
        return enhanced_output

    def generate_initial_solution_grasp(self, data, p=0.05, max_time=60):
        """Generate initial solution using GRASP (Greedy Randomized Adaptive Search Procedure)"""
        Library._id_counter = 0
        
        best_solution = None
        best_score = 0
        start_time = time.time()
        
        # Try multiple GRASP iterations within time limit
        while (time.time() - start_time) < max_time:
            # Build greedy randomized solution
            solution = self.build_grasp_solution(data, p)
            
            if solution.fitness_score > best_score:
                best_solution = solution
                best_score = solution.fitness_score
        
        return best_solution if best_solution else self.build_grasp_solution(data, p)

    def build_grasp_solution(self, data, p=0.05):
        """Build a single GRASP solution"""
        import random
        
        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0
        
        # Create candidate list of libraries sorted by efficiency
        candidates = []
        for lib in data.libs:
            if lib.signup_days < data.num_days:
                # Calculate efficiency metric
                total_score = sum(data.scores[book.id] for book in lib.books)
                time_efficiency = total_score / (lib.signup_days + 1)
                book_efficiency = total_score / len(lib.books) if lib.books else 0
                efficiency = time_efficiency + book_efficiency
                candidates.append((lib, efficiency))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        while candidates and curr_time < data.num_days:
            # Restricted Candidate List (RCL)
            rcl_size = max(1, int(len(candidates) * p))
            rcl = candidates[:rcl_size]
            
            # Randomly select from RCL
            selected_lib, _ = random.choice(rcl)
            
            # Remove from candidates
            candidates = [(lib, eff) for lib, eff in candidates if lib.id != selected_lib.id]
            
            # Check if library can be signed up
            if curr_time + selected_lib.signup_days >= data.num_days:
                unsigned_libraries.append(selected_lib.id)
                continue
            
            time_left = data.num_days - (curr_time + selected_lib.signup_days)
            max_books_scanned = time_left * selected_lib.books_per_day
            
            # Select best books not yet scanned
            available_books = []
            for book in selected_lib.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
            
            # Sort by score and take the best ones
            available_books.sort(key=lambda b: data.scores[b], reverse=True)
            selected_books = available_books[:max_books_scanned]
            
            if selected_books:
                signed_libraries.append(selected_lib.id)
                scanned_books_per_library[selected_lib.id] = selected_books
                scanned_books.update(selected_books)
                curr_time += selected_lib.signup_days
            else:
                unsigned_libraries.append(selected_lib.id)
        
        # Add remaining libraries to unsigned
        for lib, _ in candidates:
            unsigned_libraries.append(lib.id)
        
        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
        solution.calculate_fitness_score(data.scores)
        return solution

    def tweak_solution_swap_signed(self, solution, data):
        """Randomly swaps two libraries within the signed libraries list."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)
        idx1, idx2 = random.sample(range(len(solution.signed_libraries)), 2)
        
        lib_id1 = solution.signed_libraries[idx1]
        lib_id2 = solution.signed_libraries[idx2]
        
        new_signed_libraries = solution.signed_libraries.copy()
        new_signed_libraries[idx1] = lib_id2
        new_signed_libraries[idx2] = lib_id1
        
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        
        for lib_id in new_signed_libraries:
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue
            
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = []
            for book in library.books:
                if book.id not in scanned_books and len(available_books) < max_books_scanned:
                    available_books.append(book.id)
            
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)
        
        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_same_books(self, solution, data):
        """Swap books between libraries that have the same books."""
        new_solution = copy.deepcopy(solution)
        
        if len(new_solution.signed_libraries) < 2:
            return new_solution
        
        # Find libraries with overlapping books
        lib1_id, lib2_id = random.sample(new_solution.signed_libraries, 2)
        
        lib1_books = set(book.id for book in data.libs[lib1_id].books)
        lib2_books = set(book.id for book in data.libs[lib2_id].books)
        
        common_books = lib1_books & lib2_books
        
        if not common_books:
            return new_solution
        
        # Get currently scanned books for each library
        lib1_scanned = new_solution.scanned_books_per_library.get(lib1_id, [])
        lib2_scanned = new_solution.scanned_books_per_library.get(lib2_id, [])
        
        # Find swappable books
        lib1_common_scanned = [b for b in lib1_scanned if b in common_books]
        lib2_common_scanned = [b for b in lib2_scanned if b in common_books]
        
        if lib1_common_scanned and lib2_common_scanned:
            # Perform swap
            book1 = random.choice(lib1_common_scanned)
            book2 = random.choice(lib2_common_scanned)
            
            # Update scanned books
            if book1 in lib1_scanned:
                idx1 = lib1_scanned.index(book1)
                lib1_scanned[idx1] = book2
            
            if book2 in lib2_scanned:
                idx2 = lib2_scanned.index(book2)
                lib2_scanned[idx2] = book1
            
            new_solution.scanned_books_per_library[lib1_id] = lib1_scanned
            new_solution.scanned_books_per_library[lib2_id] = lib2_scanned
            
            # Update global scanned books set
            new_solution.scanned_books = set()
            for books in new_solution.scanned_books_per_library.values():
                new_solution.scanned_books.update(books)
            
            new_solution.calculate_fitness_score(data.scores)
        
        return new_solution

    def tweak_solution_insert_library(self, solution, data, target_lib=None):
        """Insert a library from unsigned to signed list."""
        new_solution = copy.deepcopy(solution)
        
        if not new_solution.unsigned_libraries:
            return new_solution
        
        # Select library to insert
        if target_lib is not None and target_lib in new_solution.unsigned_libraries:
            lib_to_insert = target_lib
        else:
            lib_to_insert = random.choice(new_solution.unsigned_libraries)
        
        # Remove from unsigned
        new_solution.unsigned_libraries.remove(lib_to_insert)
        
        # Find best position to insert
        best_position = len(new_solution.signed_libraries)
        best_score = 0
        
        for pos in range(len(new_solution.signed_libraries) + 1):
            # Try inserting at this position
            test_signed = new_solution.signed_libraries[:pos] + [lib_to_insert] + new_solution.signed_libraries[pos:]
            
            # Simulate execution with this order
            curr_time = 0
            temp_score = 0
            temp_scanned = set()
            
            for lib_id in test_signed:
                library = data.libs[lib_id]
                
                if curr_time + library.signup_days >= data.num_days:
                    break
                
                time_left = data.num_days - (curr_time + library.signup_days)
                max_books = time_left * library.books_per_day
                
                available_books = [book.id for book in library.books 
                                 if book.id not in temp_scanned][:max_books]
                
                temp_score += sum(data.scores[book_id] for book_id in available_books)
                temp_scanned.update(available_books)
                curr_time += library.signup_days
            
            if temp_score > best_score:
                best_score = temp_score
                best_position = pos
        
        # Insert at best position
        new_solution.signed_libraries.insert(best_position, lib_to_insert)
        
        # Recalculate solution
        curr_time = 0
        new_scanned_books_per_library = {}
        new_scanned_books = set()
        
        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue
            
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [book.id for book in library.books 
                             if book.id not in new_scanned_books][:max_books]
            
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
        
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_last_book(self, solution, data):
        """Swap the last book in one library with a book from another library."""
        new_solution = copy.deepcopy(solution)
        
        if len(new_solution.signed_libraries) < 2:
            return new_solution
        
        # Select two random libraries
        lib1_id, lib2_id = random.sample(new_solution.signed_libraries, 2)
        
        lib1_books = new_solution.scanned_books_per_library.get(lib1_id, [])
        lib2_books = new_solution.scanned_books_per_library.get(lib2_id, [])
        
        if not lib1_books or not lib2_books:
            return new_solution
        
        # Get last book from lib1 and random book from lib2
        last_book_lib1 = lib1_books[-1]
        random_book_lib2 = random.choice(lib2_books)
        
        # Check if swap is valid (books exist in target libraries)
        lib1_available = set(book.id for book in data.libs[lib1_id].books)
        lib2_available = set(book.id for book in data.libs[lib2_id].books)
        
        if random_book_lib2 in lib1_available and last_book_lib1 in lib2_available:
            # Perform swap
            lib1_books[-1] = random_book_lib2
            lib2_books[lib2_books.index(random_book_lib2)] = last_book_lib1
            
            new_solution.scanned_books_per_library[lib1_id] = lib1_books
            new_solution.scanned_books_per_library[lib2_id] = lib2_books
            
            # Update global scanned books
            new_solution.scanned_books = set()
            for books in new_solution.scanned_books_per_library.values():
                new_solution.scanned_books.update(books)
            
            new_solution.calculate_fitness_score(data.scores)
        
        return new_solution

    def tweak_solution_swap_signed_with_unsigned(self, solution, data, bias_type=None, bias_ratio=2/3):
        """Swap a signed library with an unsigned library."""
        new_solution = copy.deepcopy(solution)
        
        if not new_solution.signed_libraries or not new_solution.unsigned_libraries:
            return new_solution
        
        # Select libraries to swap
        signed_lib = random.choice(new_solution.signed_libraries)
        unsigned_lib = random.choice(new_solution.unsigned_libraries)
        
        # Remove from current lists
        new_solution.signed_libraries.remove(signed_lib)
        new_solution.unsigned_libraries.remove(unsigned_lib)
        
        # Add to opposite lists
        new_solution.signed_libraries.append(unsigned_lib)
        new_solution.unsigned_libraries.append(signed_lib)
        
        # Remove books from previously signed library
        if signed_lib in new_solution.scanned_books_per_library:
            del new_solution.scanned_books_per_library[signed_lib]
        
        # Recalculate solution
        curr_time = 0
        new_scanned_books_per_library = {}
        new_scanned_books = set()
        
        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                continue
            
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [book.id for book in library.books 
                             if book.id not in new_scanned_books][:max_books]
            
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
        
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_neighbor_libraries(self, solution, data):
        """Swap adjacent libraries in the signed libraries list."""
        new_solution = copy.deepcopy(solution)
        
        if len(new_solution.signed_libraries) < 2:
            return new_solution
        
        # Select random adjacent pair
        idx = random.randint(0, len(new_solution.signed_libraries) - 2)
        
        # Swap adjacent libraries
        new_solution.signed_libraries[idx], new_solution.signed_libraries[idx + 1] = \
            new_solution.signed_libraries[idx + 1], new_solution.signed_libraries[idx]
        
        # Recalculate solution with new order
        curr_time = 0
        new_scanned_books_per_library = {}
        new_scanned_books = set()
        
        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                continue
            
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [book.id for book in library.books 
                             if book.id not in new_scanned_books][:max_books]
            
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
        
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def hill_climbing_combined(self, data, iterations=1000):
        """Combined hill climbing using multiple tweak methods."""
        current_solution = self.generate_initial_solution_grasp(data)
        current_score = current_solution.fitness_score
        
        for i in range(iterations):
            # Choose random tweak method
            method = random.choice([
                self.tweak_solution_swap_signed,
                self.tweak_solution_swap_same_books,
                self.tweak_solution_insert_library
            ])
            
            neighbor = method(current_solution, data)
            
            if neighbor.fitness_score > current_score:
                current_solution = neighbor
                current_score = neighbor.fitness_score
        
        return current_score, current_solution

    def perturb_solution(self, solution, data):
        """Perturb solution for restart in great deluge algorithm."""
        perturbed = copy.deepcopy(solution)
        
        # Apply multiple random perturbations
        for _ in range(random.randint(3, 7)):
            method = random.choice([
                self.tweak_solution_swap_signed,
                self.tweak_solution_swap_same_books,
                self.tweak_solution_insert_library,
                self.tweak_solution_swap_signed_with_unsigned
            ])
            perturbed = method(perturbed, data)
        
        return perturbed