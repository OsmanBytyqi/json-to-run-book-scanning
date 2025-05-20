#!/usr/bin/env python3
"""
Great Deluge Algorithm Parameter Tuning
=======================================

Unified script supporting both fast and enhanced execution modes.

Usage:
    python3 gda_tuning.py --fast       # Fast mode (16 combinations, 3-8 minutes)
    python3 gda_tuning.py --enhanced   # Enhanced mode (264 combinations, 15-30 minutes)

University of Prishtina - Master's Course: Nature-Inspired Algorithms
Google HashCode 2020 Book Scanning Problem
"""

import argparse
import time
from models import Parser, Solver
import os
import sys
import json
import gc  # Add garbage collection
import psutil  # Add for memory monitoring

def _get_problem_category(num_books, num_libs):
    """Determine problem category based on size"""
    if num_books < 1000 and num_libs < 100:
        return "small_instance"
    elif num_books >= 10000 or num_libs >= 1000:
        return "large_instance"
    else:
        return "medium_instance"


def run_gda_tuning(mode):
    """
    Execute GDA parameter tuning in the specified mode for all files in old-instances folder
    """
    import glob
    import os
    from datetime import datetime
    
    # Mode configuration
    mode_configs = {
        "fast": {
            "config_file": "fast_gda_config.json",
            "title": "⚡ Fast Great Deluge Algorithm Parameter Tuning",
            "description": "High-speed optimized version focusing on rapid execution",
            "expected_time": "3-8 minutes per file",
            "combinations": "~16"
        },
        "enhanced": {
            "config_file": "enhanced_gda_config.json", 
            "title": "🚀 Enhanced Great Deluge Algorithm Parameter Tuning",
            "description": "Comprehensive analysis with adaptive intelligence",
            "expected_time": "15-30 minutes per file",
            "combinations": "~264"
        }
    }
    
    config = mode_configs[mode]
    
    print(config["title"])
    print("=" * len(config["title"]))
    
    # Get all .txt files from both input folders
    # input_folders = ["./input/old-instances/", "./input/new-instances/"]
    input_folders = ["./input/new-instances/batch-2/"]
    all_files = [("input/new-instances/B95k_L2k_D28.txt", "new-instances")]
    
    # 🎯 SINGLE FILE TESTING: Uncomment the line below and specify your file
    # txt_files = ["input/new-instances-batched/batch-2/SPECIFIC_FILE_NAME.txt"]
    
    # for input_folder in input_folders:
    #     if os.path.exists(input_folder):
    #         # Comment out the line below when testing single files
    #         txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    #         # 🎯 SINGLE FILE TESTING: Uncomment and modify the line below for single file
    #         # txt_files = ["input/new-instances-batched/batch-2/B1000k_L115_D230.in"]
            
    #         for file_path in txt_files:
    #             # Determine folder type (old-instances or new-instances)
    #             folder_type = "old-instances" if "old-instances" in file_path else "new-instances"
    #             all_files.append((file_path, folder_type))
    
    if not all_files:
        print(f"❌ No .txt files found in input folders")
        return
    
    print(f"📁 Found {len(all_files)} instance files to process:")
    for i, (file_path, folder_type) in enumerate(all_files, 1):
        filename = os.path.basename(file_path)
        print(f"   {i}. {filename} ({folder_type})")
    
    # Create organized directory structure
    for folder_type in ["old-instances", "new-instances"]:
        # Create result-jsons directories
        results_dir = os.path.join("result-jsons", folder_type, mode)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create output directories for solutions
        output_dir = os.path.join("output", "jsontorun", folder_type, mode)
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔧 {mode.title()} Mode Configuration:")
    print(f"   • Config file: {config['config_file']}")
    print(f"   • Results directories: result-jsons/{{old-instances,new-instances}}/{mode}/")
    print(f"   • Solution directories: output/jsontorun/{{old-instances,new-instances}}/{mode}/")
    print(f"   • Expected time per file: {config['expected_time']}")
    print(f"   • Combinations per file: {config['combinations']}")
    print(f"   • Description: {config['description']}")
    
    # Initialize solver
    solver = Solver()
    
    # Summary data for all files
    all_results_summary = []
    total_start_time = time.time()
    
    print(f"\n🚀 Starting {mode.title()} GDA Parameter Tuning for {len(all_files)} files...")
    print("=" * 80)
    
    for file_idx, (input_file, folder_type) in enumerate(all_files, 1):
        filename = os.path.basename(input_file)
        print(f"\n📖 Processing file {file_idx}/{len(all_files)}: {filename} ({folder_type})")
        
        # Add memory monitoring and cleanup
        if file_idx > 1:
            # Force garbage collection between files
            print(f"   🧹 Cleaning up memory (file {file_idx})...")
            gc.collect()
            
            # Monitor memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            if memory_usage > 80:
                print(f"   ⚠️  High memory usage: {memory_usage:.1f}% - forcing cleanup")
                gc.collect()
                time.sleep(2)  # Brief cooldown
        
        try:
            # Parse the input file
            parser = Parser(input_file)
            data = parser.parse()
            
            # Adaptive configuration based on instance size
            file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
            books_count = data.num_books
            
            print(f"   • Books: {data.num_books:,}")
            print(f"   • Libraries: {data.num_libs:,}")
            print(f"   • Scan days: {data.num_days:,}")
            print(f"   • File size: {file_size_mb:.1f}MB")
            
            # Adaptive timeout and thread management
            if books_count > 500000:  # Massive instances (500k+ books)
                timeout_multiplier = 3.0
                max_threads_override = 8  # Reduce threads for huge instances
                print(f"   🚨 Massive instance detected - using {max_threads_override} threads, {timeout_multiplier}x timeout")
            elif books_count > 100000:  # Large instances (100k+ books)  
                timeout_multiplier = 2.0
                max_threads_override = 12
                print(f"   ⚡ Large instance detected - using {max_threads_override} threads, {timeout_multiplier}x timeout")
            elif books_count > 10000:   # Medium instances
                timeout_multiplier = 1.5
                max_threads_override = 16
                print(f"   📊 Medium instance detected - using {max_threads_override} threads, {timeout_multiplier}x timeout")
            else:  # Small instances
                timeout_multiplier = 1.0
                max_threads_override = None
                print(f"   📖 Small instance - using default configuration")
            
            # Generate organized output paths
            base_filename = os.path.splitext(filename)[0]
            results_dir = os.path.join("result-jsons", folder_type, mode)
            output_file = os.path.join(results_dir, f"{base_filename}_{mode}_results.json")
            
            solution_dir = os.path.join("output", "jsontorun", folder_type, mode)
            os.makedirs(solution_dir, exist_ok=True)
            solution_file = os.path.join(solution_dir, filename)
            
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"   • Folder type: {folder_type}")
            print(f"   • Output JSON: {output_file}")
            
            file_start_time = time.time()
            
            # Create solver with adaptive configuration
            solver = Solver()
            
            # Override configuration for large instances
            if max_threads_override:
                # Temporarily modify the config for this instance
                original_config = config.copy()
                config['max_threads'] = max_threads_override
                config['timeout_multiplier'] = timeout_multiplier
            
            results = solver.run_parallel_gda_from_json(
                config_file=config['config_file'],
                data=data,
                output_file=output_file,
                input_file=input_file,
                mode=mode,
                folder_type=folder_type,
                max_threads_override=max_threads_override
            )
            
            # Restore original config
            if max_threads_override:
                config = original_config
            
            file_end_time = time.time()
            file_execution_time = file_end_time - file_start_time
            
            if results:
                # Extract summary information
                metadata = results.get('metadata', {})
                summary = results.get('summary', {})
                best_score = summary.get('best_score', 0)
                best_combination = summary.get('best_combination', 'N/A')
                
                # Store summary for this file
                file_summary = {
                    'filename': filename,
                    'folder_type': folder_type,
                    'mode': mode,
                    'best_score': best_score,
                    'best_combination': best_combination,
                    'execution_time': round(file_execution_time, 2),
                    'execution_time_formatted': f"{file_execution_time / 60:.1f} minutes",
                    'books': data.num_books,
                    'libraries': data.num_libs,
                    'days': data.num_days,
                    'file_size_mb': round(file_size_mb, 1),
                    'threads_used': max_threads_override or config.get('max_threads', 16),
                    'timeout_multiplier': timeout_multiplier,
                    'problem_category': _get_problem_category(data.num_books, data.num_libs)
                }
                
                all_results_summary.append(file_summary)
                
                # Save detailed results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"   ✅ Completed successfully!")
                print(f"   ⏱️  Execution time: {file_execution_time:.2f} seconds ({file_execution_time / 60:.1f} minutes)")
                print(f"   🏆 Best score: {best_score:,}")
                print(f"   🥇 Best combination: {best_combination}")
                print(f"   💾 Solution saved: {solution_file}")
                print("-" * 80)
                
                # Memory cleanup after successful processing
                del data, parser, solver, results
                gc.collect()
                
            else:
                print(f"   ❌ Failed to process {filename}")
                # Still add to summary with 0 score
                books_count = getattr(data, 'num_books', 0)
                libs_count = getattr(data, 'num_libs', 0)
                file_summary = {
                    'filename': filename,
                    'folder_type': folder_type,
                    'mode': mode,
                    'best_score': 0,
                    'best_combination': 'FAILED',
                    'execution_time': round(file_execution_time, 2),
                    'execution_time_formatted': f"{file_execution_time / 60:.1f} minutes",
                    'books': books_count,
                    'libraries': libs_count,
                    'days': getattr(data, 'num_days', 0),
                    'file_size_mb': round(file_size_mb, 1),
                    'threads_used': max_threads_override or config.get('max_threads', 16),
                    'timeout_multiplier': timeout_multiplier,
                    'problem_category': _get_problem_category(books_count, libs_count)
                }
                all_results_summary.append(file_summary)
                
        except FileNotFoundError:
            print(f"   ❌ File not found: {input_file}")
            file_summary = {
                'filename': filename,
                'folder_type': folder_type,
                'mode': mode,
                'best_score': 0,
                'best_combination': 'FILE_NOT_FOUND',
                'execution_time': 0,
                'execution_time_formatted': 'N/A',
                'books': 0,
                'libraries': 0,
                'days': 0,
                'file_size_mb': round(file_size_mb, 1) if 'file_size_mb' in locals() else 0,
                'threads_used': max_threads_override or config.get('max_threads', 16) if 'max_threads_override' in locals() else 16,
                'timeout_multiplier': timeout_multiplier if 'timeout_multiplier' in locals() else 1.0,
                'problem_category': _get_problem_category(0, 0)
            }
            all_results_summary.append(file_summary)
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {str(e)}")
            file_summary = {
                'filename': filename,
                'folder_type': folder_type,
                'mode': mode,
                'best_score': 0,
                'best_combination': f'ERROR: {str(e)[:50]}',
                'execution_time': 0,
                'execution_time_formatted': 'N/A',
                'books': 0,
                'libraries': 0,
                'days': 0,
                'file_size_mb': round(file_size_mb, 1) if 'file_size_mb' in locals() else 0,
                'threads_used': max_threads_override or config.get('max_threads', 16) if 'max_threads_override' in locals() else 16,
                'timeout_multiplier': timeout_multiplier if 'timeout_multiplier' in locals() else 1.0,
                'problem_category': _get_problem_category(0, 0)
            }
            all_results_summary.append(file_summary)
            # Force cleanup on error
            gc.collect()
            time.sleep(1)  # Brief pause before next file
        
        print("-" * 80)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # Create comprehensive summary text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"{mode}_gda_summary_{timestamp}.txt"
    
    print(f"\n📊 Creating comprehensive summary: {summary_filename}")
    
    with open(summary_filename, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write(f"GREAT DELUGE ALGORITHM - {mode.upper()} MODE SUMMARY REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {mode.title()}\n")
        f.write(f"Config File: {config['config_file']}\n")
        f.write(f"Total Files Processed: {len(all_files)}\n")
        f.write(f"Total Execution Time: {total_execution_time:.2f} seconds ({total_execution_time/60:.1f} minutes)\n")
        f.write("\n")
        
        # Overall Statistics
        successful_files = [r for r in all_results_summary if r['best_score'] > 0]
        failed_files = [r for r in all_results_summary if r['best_score'] == 0]
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Successful runs: {len(successful_files)}/{len(all_files)}\n")
        f.write(f"Failed runs: {len(failed_files)}/{len(all_files)}\n")
        
        if successful_files:
            total_scores = [r['best_score'] for r in successful_files]
            total_times = [r['execution_time'] for r in successful_files]
            
            f.write(f"Average best score: {sum(total_scores)/len(total_scores):,.2f}\n")
            f.write(f"Total score across all files: {sum(total_scores):,}\n")
            f.write(f"Highest score: {max(total_scores):,}\n")
            f.write(f"Lowest score: {min(total_scores):,}\n")
            f.write(f"Average execution time: {sum(total_times)/len(total_times):.2f} seconds\n")
            f.write(f"Total computation time: {sum(total_times):.2f} seconds\n")
        
        f.write("\n")
        
        # Detailed Results Table
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Rank':<4} {'Filename':<25} {'Folder':<15} {'Score':<15} {'Time (s)':<10} {'Time (m)':<10} {'Best Combination':<25}\n")
        f.write("-" * 110 + "\n")
        
        # Sort by best score (descending)
        sorted_results = sorted(all_results_summary, key=lambda x: x['best_score'], reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            filename = result['filename'][:24]  # Truncate if too long
            folder_type = result['folder_type'][:14]  # Truncate if too long
            score = f"{result['best_score']:,}"
            time_s = f"{result['execution_time']:.1f}"
            time_m = f"{result['execution_time']/60:.2f}"
            combination = result['best_combination'][:24]  # Truncate if too long
            
            f.write(f"{rank:<4} {filename:<25} {folder_type:<15} {score:<15} {time_s:<10} {time_m:<10} {combination:<25}\n")
        
        f.write("\n")
        
        # Problem Size Analysis
        f.write("PROBLEM SIZE ANALYSIS:\n")
        f.write("-" * 115 + "\n")
        f.write(f"{'Filename':<25} {'Folder':<15} {'Books':<10} {'Libraries':<12} {'Days':<6} {'Category':<15} {'Score':<15}\n")
        f.write("-" * 115 + "\n")
        
        for result in sorted_results:
            filename = result['filename'][:24]
            folder_type = result['folder_type'][:14]
            books = f"{result['books']:,}"
            libraries = f"{result['libraries']:,}"
            days = str(result['days'])
            category = result.get('problem_category', 'unknown')[:14]
            score = f"{result['best_score']:,}"
            
            f.write(f"{filename:<25} {folder_type:<15} {books:<10} {libraries:<12} {days:<6} {category:<15} {score:<15}\n")
        
        f.write("\n")
        
        # Performance Analysis by Category
        if successful_files:
            f.write("PERFORMANCE BY PROBLEM CATEGORY:\n")
            f.write("-" * 50 + "\n")
            
            categories = {}
            for result in successful_files:
                cat = result['problem_category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result)
            
            for category, results in categories.items():
                scores = [r['best_score'] for r in results]
                times = [r['execution_time'] for r in results]
                
                f.write(f"{category.title()} Instances ({len(results)} files):\n")
                f.write(f"  Average Score: {sum(scores)/len(scores):,.2f}\n")
                f.write(f"  Best Score: {max(scores):,}\n")
                f.write(f"  Average Time: {sum(times)/len(times):.2f} seconds\n")
                f.write(f"  Total Time: {sum(times):.2f} seconds\n")
                f.write("\n")
        
        # Footer
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    # Display final summary to console
    print(f"\n🎉 {mode.title()} GDA Parameter Tuning Complete!")
    print("=" * 80)
    print(f"📊 FINAL SUMMARY:")
    print(f"   • Total files processed: {len(all_files)}")
    print(f"   • Successful runs: {len(successful_files)}")
    print(f"   • Failed runs: {len(failed_files)}")
    print(f"   • Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.1f} minutes)")
    
    # Count files by folder type
    old_instances_count = len([f for f in all_files if f[1] == 'old-instances'])
    new_instances_count = len([f for f in all_files if f[1] == 'new-instances'])
    
    print(f"   • Old instances: {old_instances_count} files")
    print(f"   • New instances: {new_instances_count} files")
    
    if successful_files:
        best_result = max(all_results_summary, key=lambda x: x['best_score'])
        print(f"   • Best overall score: {best_result['best_score']:,} (from {best_result['filename']} in {best_result['folder_type']})")
        print(f"   • Total combined score: {sum(r['best_score'] for r in successful_files):,}")
    
    print(f"\n💾 Files Created:")
    print(f"   • Summary report: {summary_filename}")
    print(f"   • JSON results organized in:")
    if old_instances_count > 0:
        print(f"     - result-jsons/old-instances/{mode}/ ({len([r for r in all_results_summary if r['best_score'] > 0 and r['folder_type'] == 'old-instances'])} files)")
    if new_instances_count > 0:
        print(f"     - result-jsons/new-instances/{mode}/ ({len([r for r in all_results_summary if r['best_score'] > 0 and r['folder_type'] == 'new-instances'])} files)")
    print(f"   • Solution files organized in:")
    if old_instances_count > 0:
        print(f"     - output/jsontorun/old-instances/{mode}/ (best solutions)")
    if new_instances_count > 0:
        print(f"     - output/jsontorun/new-instances/{mode}/ (best solutions)")
    
    if mode == "fast":
        expected_slow_time = total_execution_time * 4  # Estimate 4x slower for enhanced
        print(f"\n⚡ Speed Analysis:")
        print(f"   • Average time per file: {total_execution_time/len(all_files):.1f} seconds")
        print(f"   • Estimated enhanced mode time: ~{expected_slow_time/60:.1f} minutes")
    else:
        print(f"\n📚 Enhanced Analysis:")
        print(f"   • Average time per file: {total_execution_time/len(all_files):.1f} seconds")
        print(f"   • Comprehensive parameter analysis completed")
    
    print(f"\n🔬 Research Output Complete!")
    print(f"{'⚡ Production-ready fast optimization' if mode == 'fast' else '📚 Research-grade comprehensive analysis'}")

def main():
    """
    Main execution function with argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Great Deluge Algorithm Parameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 gda_tuning.py --fast       # Fast mode (16 combinations, 3-8 minutes)
  python3 gda_tuning.py --enhanced   # Enhanced mode (264 combinations, 15-30 minutes)
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--fast', action='store_true', 
                           help='Run fast mode (16 combinations, optimized for speed)')
    mode_group.add_argument('--enhanced', action='store_true',
                           help='Run enhanced mode (264 combinations, comprehensive analysis)')
    
    args = parser.parse_args()
    
    # Determine mode
    mode = "fast" if args.fast else "enhanced"
    
    print(f"Great Deluge Algorithm Parameter Tuning System")
    print(f"University of Prishtina - Nature-Inspired Algorithms")
    print(f"Google HashCode 2020 Book Scanning Problem")
    print(f"{'=' * 60}")
    
    # Show mode-specific features
    analyze_mode_features(mode)
    
    # Run the tuning
    run_gda_tuning(mode)
    
    print(f"\n🎉 {mode.title()} GDA Parameter Tuning Complete!")
    if mode == "fast":
        print(f"⚡ Optimized for speed while maintaining solution quality")
        print(f"🔬 Production-ready fast metaheuristic optimization")
    else:
        print(f"📚 State-of-the-art adaptive metaheuristic optimization")
        print(f"🔬 Research-grade comprehensive parameter analysis")

if __name__ == "__main__":
    main() 
