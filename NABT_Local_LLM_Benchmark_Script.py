import os
import re
import time
import json
import signal
import platform
import sys
import subprocess
import pandas as pd
import numpy as np
import select
import matplotlib.pyplot as plt
from datetime import datetime
import gc # Explicitly import garbage collector

# Attempt to import rich and other core dependencies early for UI/error handling
try:
    from rich.console import Console
    import rich.box
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy Console if rich is missing, so basic prints work
    class DummyConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("-" * 60)
        def clear(self): os.system('cls' if os.name == 'nt' else 'clear')
    console = DummyConsole()
    print("Warning: 'rich' library not found. Formatting will be basic.")

# Attempt to import other dependencies, check later
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False # We check this later in the dependency check

# Torch check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Initialize Console (might be dummy if rich failed)
console = Console()

# Default settings
DEFAULT_CONFIG = {
    "context_length": 4096,
    "timeout_seconds": 90,
    "batch_size": 2024,
    "max_tokens": 2024,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "echo": False,
    "checkpoint_interval": 50
}

def detect_hardware():
    """Detect available hardware (CPU, Apple Silicon/Metal, CUDA)"""
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
    cpu_count = os.cpu_count() or 1
    perf_cores = max(cpu_count // 2, 1) if is_apple_silicon else max(cpu_count - 2, 1)

    cuda_available = False
    cuda_device_name = "N/A"
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                 cuda_available = True
                 cuda_device_name = torch.cuda.get_device_name(0)
        except Exception: cuda_available = False

    gpu_available = is_apple_silicon or cuda_available
    gpu_type = "Apple Metal" if is_apple_silicon else ("NVIDIA CUDA" if cuda_available else "None")

    return {
        "cpu_count": cpu_count, "recommended_threads": perf_cores,
        "is_apple_silicon": is_apple_silicon, "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name, "gpu_available": gpu_available,
        "gpu_type": gpu_type
    }

# Assume 'console' and 'RICH_AVAILABLE' are defined appropriately elsewhere
# Example:
try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    # Basic console printing fallback
    class SimpleConsole:
        def print(self, text=""): print(text)
        def clear(self): import os; os.system('cls' if os.name == 'nt' else 'clear')
    console = SimpleConsole()

# --- Project & Citation Information ---
PROJECT_TITLE = "The Neurosurgeon’s Introduction to Open-Source Artificial Intelligence: A Downloadable Pipeline for Neurosurgical Benchmarking of Open-Source and Cloud-Based Large Language Models"
AUTHORS = "David Gomez, BS¹; Ishan Shah, BS¹; Richard Hislop, BS²; Benjamin Hopkins, MD¹; Gage Guerrera, BS¹; David J. Cote, MD PhD¹; Lawrance K. Chung MD¹; Robert G. Briggs, MD¹; William Mack MD¹; Gabriel Zada, MD¹"
AFFILIATIONS = "¹Department of Neurosurgery, Keck School of Medicine, University of Southern California, Los Angeles, CA"
JOURNAL_TARGET = "JNS Focus 2025"
CORRESPONDING_AUTHOR = "David Gomez, BS"
CORRESPONDING_EMAIL = "gomezdav@usc.edu"
# --- End Information ---

def display_welcome_screen():
    """Display a welcome screen for the NABT Local LLM Benchmarking script (1/3)."""
    console.clear()

    # --- Formatted Citation Strings ---
    citation_info_rich = (
        f"[bold]Reference:[/bold]\n"
        f"{AUTHORS}. ({JOURNAL_TARGET}).\n"
        f"'{PROJECT_TITLE}'.\n" # Title in quotes
        f"{AFFILIATIONS}.\n\n"
        f"[bold]Contact:[/bold] Inquiries regarding this toolkit may be directed to {CORRESPONDING_AUTHOR} ({CORRESPONDING_EMAIL})."
    )
    citation_info_plain = (
        f"Reference:\n"
        f"{AUTHORS}. ({JOURNAL_TARGET}).\n"
        f"'{PROJECT_TITLE}'.\n"
        f"{AFFILIATIONS}.\n\n"
        f"Contact: Inquiries regarding this toolkit may be directed to {CORRESPONDING_AUTHOR} ({CORRESPONDING_EMAIL})."
    )

    if RICH_AVAILABLE:
        # --- Main Content Panel ---
        main_panel_content = (
            "[bold blue]NABT: Local LLM Benchmarking (Script 1/3)[/bold blue]\n\n"
            "This script executes quantitative benchmarking of Large Language Models (LLMs) "
            "deployed locally in GGUF format. No internet connection is needed.\n\n"
            "Model performance is evaluated against any CSV question set desired.\n"
            " • Response Accuracy is determined by automated parsing with optional manual review.\n"
            " • Inference time in (s) per Question is collected \n\n"
            "Utilizes the llama-cpp-python library for hardware-accelerated inference (CPU/GPU), "
        )
        console.print(Panel.fit(
            main_panel_content,
            title="Neurosurgery AI Benchmarking Toolkit - Local Evaluation",
            border_style="blue",
            padding=(1, 2)
        ))

        console.print() # Spacer

        # --- Citation Panel ---
        console.print(Panel.fit(
            citation_info_rich,
            title="Project Information & Citation",
            border_style="grey70", # Use a subtle border for citation
            padding=(1, 2)
        ))

    else:
        # Fallback for environments without Rich
        console.print("--- NABT: Local LLM Benchmarking (Script 1/3) ---")
        console.print("\nExecutes quantitative benchmarking of locally deployed LLMs (GGUF format).")
        console.print("Evaluates performance (Accuracy, Inference Time) using a dataset of your choosing.")
        console.print("Utilizes llama-cpp-python for CPU/GPU acceleration. No internet connection needed.")
        console.print("\n----------------------------------------------------")
        console.print(f"\n{citation_info_plain}") # Print plain citation info
        console.print("\n----------------------------------------------------")

    console.print() # Add a final blank line

# Example usage (ensure console and RICH_AVAILABLE are set):
# display_welcome_screen()

    hardware_info = detect_hardware()

    if RICH_AVAILABLE:
        table = Table(title="[bold magenta]System Information[/bold magenta]", show_header=False, box=rich.box.ROUNDED)
        table.add_column("Component", style="cyan", justify="right")
        table.add_column("Details", style="green")
        table.add_row("CPU:", f"{hardware_info['cpu_count']} cores (Recommended threads: {hardware_info['recommended_threads']})")
        if hardware_info["is_apple_silicon"]: table.add_row("GPU (Apple):", "Metal Acceleration Available ✓")
        elif hardware_info["cuda_available"]: table.add_row("GPU (NVIDIA):", f"CUDA Acceleration Available ✓ ({hardware_info['cuda_device_name']})")
        else: table.add_row("GPU Acceleration:", "Not Detected / Unavailable ✗")
        if not TORCH_AVAILABLE and not hardware_info["is_apple_silicon"]:
             table.add_row("[yellow]PyTorch Status:[/yellow]", "[yellow]Not Found (Required for CUDA detection/use)[/yellow]")
        console.print(table)
    else: # Basic print if rich is missing
        console.print("System Information:")
        console.print(f"- CPU: {hardware_info['cpu_count']} cores (Recommended threads: {hardware_info['recommended_threads']})")
        gpu_status = "N/A"
        if hardware_info["is_apple_silicon"]: gpu_status = "Metal Available"
        elif hardware_info["cuda_available"]: gpu_status = f"CUDA Available ({hardware_info['cuda_device_name']})"
        else: gpu_status = "Not Detected/Unavailable"
        console.print(f"- GPU: {gpu_status}")
        if not TORCH_AVAILABLE and not hardware_info["is_apple_silicon"]:
             console.print("- PyTorch Status: Not Found (Needed for CUDA)")
    console.print()


def get_user_input():
    """Get the necessary paths and configuration from the user"""
    config = DEFAULT_CONFIG.copy()
    hardware_info = detect_hardware()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["results_dir"] = f"./benchmark_results_{timestamp}"
    try:
        os.makedirs(config["results_dir"], exist_ok=True)
    except OSError as e:
         console.print(f"[bold red]Error creating results directory {config['results_dir']}: {e}[/bold red]")
         sys.exit(1)

    # --- Navigation Helper ---
    def handle_navigation(user_input):
        if user_input.lower() == 'exit':
            console.print("[bold yellow]Exiting program.[/bold yellow]")
            sys.exit(0)
        if user_input.lower() == 'back':
            console.print("[yellow]Going back...[/yellow]")
            return True
        return False

    # --- Input Loop ---
    current_step = 1
    while current_step <= 5:
        try:
            if current_step == 1: # CSV Path
                console.print("[bold cyan]1. Enter path to question bank CSV file (or 'exit'):[/bold cyan]")
                csv_path = input("> ").strip()
                if handle_navigation(csv_path): continue # Exit directly

                if os.path.exists(csv_path) and csv_path.endswith('.csv'):
                    config["csv_path"] = csv_path
                    try:
                        df_temp = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
                        config["total_questions"] = len(df_temp)
                        console.print(f"[bold green]✓ Loaded CSV: '{os.path.basename(csv_path)}' ({config['total_questions']} questions)[/bold green]")
                        # Keep preview simple
                        console.print(f"[bold]Headers:[/bold] {', '.join(df_temp.columns)}")
                        del df_temp; gc.collect()
                        current_step += 1
                    except Exception as e:
                        console.print(f"[bold red]Error reading/previewing CSV: {e}[/bold red]")
                else:
                    console.print("[bold red]File not found or not a .csv file.[/bold red]")

            elif current_step == 2: # Model Paths
                console.print("\n[bold cyan]2. Enter model file paths (.gguf/.bin) or folders.[/bold cyan]")
                console.print("   Type 'done', 'back' (to CSV), or 'exit'.")
                model_input = input("> ").strip()
                if handle_navigation(model_input): current_step -= 1; continue

                if model_input.lower() == 'done':
                    if config.get("model_paths"): current_step += 1; continue
                    else: console.print("[bold red]Add at least one model.[/bold red]"); continue

                added_count = 0
                if os.path.isdir(model_input):
                    try: # Add folder scanning logic
                        for file in os.listdir(model_input):
                            fp = os.path.join(model_input, file)
                            if os.path.isfile(fp) and (file.endswith('.gguf') or file.endswith('.bin')):
                                if fp not in config.setdefault("model_paths", []):
                                    config["model_paths"].append(fp)
                                    console.print(f"[green]  + Added: {os.path.basename(fp)}[/green]")
                                    added_count += 1
                        if added_count == 0: console.print(f"[yellow]No new models found in {model_input}[/yellow]")
                        else: console.print(f"[bold green]✓ Added {added_count} from folder.[/bold green]")
                    except Exception as e: console.print(f"[red]Error scanning folder: {e}[/red]")
                elif os.path.isfile(model_input) and (model_input.endswith('.gguf') or model_input.endswith('.bin')):
                    if model_input not in config.setdefault("model_paths", []):
                        config["model_paths"].append(model_input)
                        console.print(f"[bold green]✓ Added model: {os.path.basename(model_input)}[/bold green]")
                    else: console.print(f"[yellow]Model already added.[/yellow]")
                else: console.print("[bold red]Invalid path or not a supported model.[/bold red]")
                # Show current models if any added
                if config.get("model_paths"):
                     console.print(f"--- Models ({len(config['model_paths'])}): {[os.path.basename(p) for p in config['model_paths']]} ---")

            elif current_step == 3: # Number of Questions
                console.print(f"\n[bold cyan]3. Questions to test? (1-{config['total_questions']}, 'all', 'back', 'exit')[/bold cyan]")
                num_input = input("> ").strip()
                if handle_navigation(num_input): current_step -= 1; continue

                if num_input.lower() == 'all':
                    config["num_questions"] = config['total_questions']
                    console.print(f"[green]✓ Selected all {config['total_questions']}.[/green]")
                    current_step += 1
                else:
                    try:
                        num = int(num_input)
                        if 1 <= num <= config['total_questions']:
                            config["num_questions"] = num
                            console.print(f"[green]✓ Selected {num}.[/green]")
                            current_step += 1
                        else: console.print(f"[red]Number out of range.[/red]")
                    except ValueError: console.print("[red]Invalid input.[/red]")

            elif current_step == 4: # CPU Threads
                recommended_threads = hardware_info["recommended_threads"]
                console.print(f"\n[bold cyan]4. CPU threads? (Rec: {recommended_threads}, Enter=default, 'back', 'exit')[/bold cyan]")
                threads_input = input("> ").strip()
                if handle_navigation(threads_input): current_step -= 1; continue

                if not threads_input:
                    config["threads"] = recommended_threads
                    console.print(f"[green]✓ Using default {recommended_threads}.[/green]")
                    current_step += 1
                else:
                    try:
                        threads = int(threads_input)
                        if threads > 0:
                            config["threads"] = threads
                            console.print(f"[green]✓ Using {threads}.[/green]")
                            current_step += 1
                        else: console.print("[red]Enter positive number.[/red]")
                    except ValueError: console.print("[red]Invalid input.[/red]")

            elif current_step == 5: # GPU Choice
                config["use_gpu"] = False # Default
                if hardware_info["gpu_available"]:
                    console.print(f"\n[bold cyan]5. Use GPU {hardware_info['gpu_type']}? (Y/n, 'back', 'exit')[/bold cyan]")
                    gpu_input = input("> ").strip().lower()
                    if handle_navigation(gpu_input): current_step -= 1; continue

                    if gpu_input in ['y', 'yes', '']:
                        config["use_gpu"] = True
                        console.print(f"[green]✓ GPU enabled ({hardware_info['gpu_type']}).[/green]")
                        current_step += 1
                    elif gpu_input in ['n', 'no']:
                        config["use_gpu"] = False
                        console.print("[yellow]✓ GPU disabled.[/yellow]")
                        current_step += 1
                    else: console.print("[red]Invalid input.[/red]")
                else:
                    console.print("\n[yellow]GPU not detected/available. Using CPU only.[/yellow]")
                    config["use_gpu"] = False
                    current_step += 1 # Auto-advance if no GPU choice needed

        except KeyboardInterrupt: # Allow Ctrl+C during config
            console.print("\n[yellow]Configuration interrupted.[/yellow]")
            sys.exit(0)

    # --- Final Confirmation ---
    console.print("\n[bold underline]Configuration Summary:[/bold underline]")
    avg_time_per_q = 8 if config["use_gpu"] else 20
    total_models = len(config["model_paths"])
    total_q_run = config["num_questions"]
    est_sec = total_models * total_q_run * avg_time_per_q
    if est_sec >= 3600: time_disp = f"~{est_sec / 3600:.1f} hours"
    elif est_sec >= 60: time_disp = f"~{est_sec / 60:.1f} minutes"
    else: time_disp = f"~{est_sec:.0f} seconds"

    if RICH_AVAILABLE:
        summary_table = Table(show_header=False, box=rich.box.ROUNDED, border_style="blue")
        summary_table.add_column("Setting", style="cyan", justify="right", width=18)
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Question File:", os.path.basename(config["csv_path"]))
        summary_table.add_row("Models Selected:", str(total_models))
        summary_table.add_row("Questions to Run:", f"{total_q_run} / {config['total_questions']}")
        summary_table.add_row("CPU Threads:", str(config["threads"]))
        summary_table.add_row("Execution Mode:", "[green]GPU Enabled[/green]" if config["use_gpu"] else "[yellow]CPU Only[/yellow]")
        summary_table.add_row("Estimated Time:", time_disp)
        summary_table.add_row("Results Dir:", config["results_dir"])
        console.print(summary_table)
    else: # Basic print
         console.print(f"- File: {os.path.basename(config['csv_path'])}")
         console.print(f"- Models: {total_models}")
         # ... print other config items ...

    while True:
        console.print("\n[bold cyan]Proceed with benchmark? (Y/n, 'back' to restart, 'exit')[/bold cyan]")
        proceed = input("> ").strip().lower()
        if handle_navigation(proceed): return get_user_input() # Restart config

        if proceed in ['y', 'yes', '']:
            console.print("[bold green]Starting benchmark...[/bold green]\n")
            break
        elif proceed in ['n', 'no']:
            console.print("[yellow]Benchmark cancelled.[/yellow]")
            sys.exit(0)
        else: console.print("[red]Invalid input.[/red]")

    return config


# --- PROMPT AND PARSING (v4.2) ---

def build_universal_prompt(question, choices):
    """Create a standardized prompt requiring a specific final answer phrase."""
    formatted_choices = choices
    if isinstance(choices, str):
        choice_lines = [line.strip() for line in choices.strip().split('\n') if line.strip()]
        formatted_count = sum(1 for line in choice_lines if re.match(r"^\s*[A-Ea-e][.:]\s+", line))
        is_already_formatted = formatted_count >= len(choice_lines) / 2

        if not is_already_formatted:
             formatted_lines = []
             letter_idx = 0
             for line in choice_lines:
                  cleaned_line = re.sub(r"^\s*[A-Ea-e][.:]?\s*", "", line).strip()
                  if cleaned_line:
                      formatted_lines.append(f"{chr(ord('A') + letter_idx)}. {cleaned_line}")
                      letter_idx += 1
             formatted_choices = "\n".join(formatted_lines)
        else:
             formatted_choices = "\n".join(line.strip() for line in choice_lines)

    return (
        f"**Instruction**:\n"
        f"You are a neurosurgery resident taking a multiple-choice board exam.\n"
        f"1. Carefully read the Question and Answer Choices.\n"
        f"2. Analyze the options and determine the single best answer (A, B, C, D, or E, etc).\n"
        f"3. Conclude your *entire* response with the following exact phrase on a new line: \n"
        f"   `Final Answer: The final answer is letter [X]`\n"
        f"   Replace [X] with the capital letter of your chosen answer. This phrase must be the absolute last part of your response.\n\n"
        f"**Question**:\n{question}\n\n"
        f"**Answer Choices**:\n{formatted_choices}\n\n"
        f"**Response**:"
    )


def parse_final_answer_letter(text):
    """
    Parses the final answer letter (A-E) from the end of the text using
    a tiered approach with multiple regex patterns for flexibility.
    Returns the uppercase letter if found, otherwise None.
    """
    if not isinstance(text, str):
        return None

    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    # Define patterns in order of priority (most explicit first)
    # \W* handles optional non-word chars (like ., *, `) after the letter
    # \s* handles optional whitespace before the end $
    patterns = [
        # 1. Strongest: Explicit "Final Answer: The final answer is letter [X]" variations
        re.compile(r"`?Final Answer:\s*The final answer is letter\s+([A-E])\W*`?\W*\s*$", re.IGNORECASE | re.MULTILINE),
        # 2. Strong: Explicit "Final Answer: The final answer is [X]" variations
        re.compile(r"`?Final Answer:\s*The final answer is\s+([A-E])\W*`?\W*\s*$", re.IGNORECASE | re.MULTILINE),
        # 3. Medium: "The final answer is letter [X]" variations (no "Final Answer:")
        re.compile(r"The final answer is letter\s+([A-E])\W*\s*$", re.IGNORECASE | re.MULTILINE),
        # 4. Medium: "The final answer is [X]" variations (no "Final Answer:")
        re.compile(r"The final answer is\s+([A-E])\W*\s*$", re.IGNORECASE | re.MULTILINE),
        # 5. Weaker: Simple "Answer: [X]" at the end
        re.compile(r"Answer:\s*([A-E])\W*\s*$", re.IGNORECASE | re.MULTILINE),
        # 6. Weakest: Just a letter A-E, optionally followed by punctuation, alone on the last line
        #    We check this separately on the last line only to avoid matching letters mid-reasoning.
        re.compile(r"^\s*([A-E])\W*\s*$", re.IGNORECASE) # Note: No MULTILINE, applied to last line only
    ]

    # Try patterns 1 through 5 on the whole text (anchored to the end)
    for pattern in patterns[:-1]: # Exclude the last pattern for now
        match = pattern.search(cleaned_text)
        if match:
            # print(f"DEBUG: Matched pattern {patterns.index(pattern)+1}") # Optional debug print
            return match.group(1).upper()

    # If no match yet, try pattern 6 on the LAST non-empty line
    lines = cleaned_text.splitlines()
    last_line = ""
    for line in reversed(lines):
        stripped_line = line.strip()
        if stripped_line:
            last_line = stripped_line
            break

    if last_line:
        match = patterns[-1].search(last_line) # Try the last pattern (standalone letter)
        if match:
             # print("DEBUG: Matched pattern 6 (last line)") # Optional debug print
             return match.group(1).upper()

    # If nothing matches
    # print("DEBUG: No pattern matched") # Optional debug print
    return None


def manual_review(results_df, questions_df):
    """
    Interactive manual review of incorrect answers.
    Allows users to override automatic grading for borderline cases.
    """
    if not RICH_AVAILABLE:
        console.print("[yellow]Manual review works better with rich library installed.[/yellow]")
    
    incorrect_results = results_df[
        (~results_df["is_correct"]) & 
        (~results_df["timed_out"]) & 
        (results_df["error"].isna())
    ].copy()
    
    if incorrect_results.empty:
        console.print("[green]No incorrect answers to review![/green]")
        return results_df
    
    console.print(f"\n[bold]Manual Review Mode - {len(incorrect_results)} items to review[/bold]")
    console.print("For each response marked incorrect, review and decide if it should be changed.")
    console.print("Enter Y to mark as correct, N to keep as incorrect, S to skip, Q to quit review.\n")
    
    changes_made = 0
    
    for idx, row in incorrect_results.iterrows():
        q_id = row['question_id_csv']
        q_data = questions_df.iloc[q_id]
        model_name = row['model']
        
        # Display question data
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[bold cyan]Question:[/bold cyan]\n{q_data['Question']}\n\n"
                f"[bold cyan]Choices:[/bold cyan]\n{q_data['All Answer Choices']}\n\n"
                f"[bold green]Correct Answer:[/bold green] {q_data['Correct Answer']} "
                f"(Letter: {row['correct_letter']})",
                title=f"Review Item {changes_made+1}/{len(incorrect_results)}", 
                border_style="blue"
            ))
            
            # Display model response
            console.print(Panel(
                f"{row['model_raw_response']}\n\n"
                f"[bold red]Auto-Parsed Answer:[/bold red] {row['model_answer_letter'] or 'None'}\n"
                f"[bold yellow]Expected Answer:[/bold yellow] {row['correct_letter']}",
                title=f"Response from {model_name}", 
                border_style="yellow"
            ))
        else:
            # Basic print version
            console.print(f"\n--- Review Item {changes_made+1}/{len(incorrect_results)} ---")
            console.print(f"Question: {q_data['Question']}")
            console.print(f"Choices: {q_data['All Answer Choices']}")
            console.print(f"Correct Answer: {q_data['Correct Answer']} (Letter: {row['correct_letter']})")
            console.print(f"\nResponse from {model_name}:")
            console.print(f"{row['model_raw_response'][:500]}..." if len(row['model_raw_response']) > 500 else row['model_raw_response'])
            console.print(f"Auto-Parsed: {row['model_answer_letter'] or 'None'}, Expected: {row['correct_letter']}")
        
        # Get user decision
        while True:
            decision = input("\nMark as correct? (Y/N/S/Q) or enter letter A-E to override: ").strip().upper()
            if decision in ['Y', 'N', 'S', 'Q', 'A', 'B', 'C', 'D', 'E']:
                break
            console.print("[red]Invalid input. Use Y, N, S, Q, or A-E.[/red]")
        
        if decision == 'Q':
            console.print("[yellow]Manual review aborted.[/yellow]")
            break
        elif decision == 'S':
            console.print("[cyan]Skipped.[/cyan]")
            continue
        elif decision == 'Y':
            # Update the results DataFrame
            results_df.at[idx, 'is_correct'] = True
            results_df.at[idx, 'review_note'] = "Manually marked correct"
            changes_made += 1
            console.print("[green]Marked as correct.[/green]")
        elif decision in ['A', 'B', 'C', 'D', 'E']:
            # Update with the specified letter
            results_df.at[idx, 'model_answer_letter'] = decision
            results_df.at[idx, 'is_correct'] = (decision == row['correct_letter'])
            results_df.at[idx, 'review_note'] = f"Answer manually set to {decision}"
            if decision == row['correct_letter']:
                changes_made += 1
                console.print(f"[green]Answer updated to {decision} (correct).[/green]")
            else:
                console.print(f"[cyan]Answer updated to {decision} (still incorrect).[/cyan]")
        else:  # 'N'
            results_df.at[idx, 'review_note'] = "Manually confirmed incorrect"
            console.print("[red]Kept as incorrect.[/red]")
    
    console.print(f"\n[bold]Manual review complete. {changes_made} answers were changed.[/bold]")
    return results_df


# --- Timeout Handling ---
class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException("Timeout")


# --- Model Initialization ---
def initialize_llama_model(model_path, config):
    """Initialize Llama model with hardware detection"""
    hardware_info = detect_hardware()
    model_options = {
        "model_path": model_path, "n_ctx": config["context_length"],
        "n_threads": config.get("threads", hardware_info["recommended_threads"]),
        "n_batch": config.get("batch_size", 1024),
        "use_mmap": True, "use_mlock": False,
        "verbose": LLAMA_VERBOSE
    }
    if config.get("use_gpu", False) and hardware_info["gpu_available"]:
        model_options["n_gpu_layers"] = -1
        gpu_log_msg = f"[bold green]Attempting GPU acceleration via {hardware_info['gpu_type']}[/bold green]"
        if hardware_info['cuda_available']: gpu_log_msg += f" ({hardware_info['cuda_device_name']})"
        console.print(gpu_log_msg)
    else:
        model_options["n_gpu_layers"] = 0
        reason = "by user choice" if config.get("use_gpu") and hardware_info['gpu_available'] else "GPU not detected/available"
        console.print(f"[yellow]Using CPU only ({reason}).[/yellow]")

    console.print(f"Initializing [magenta]{os.path.basename(model_path)}[/magenta] ({model_options['n_threads']} threads, {model_options['n_gpu_layers']} GPU layers)...")
    try:
        llm_instance = Llama(**model_options)
        return llm_instance
    except Exception as e:
        console.print(f"[bold red]Error initializing Llama model: {e}[/bold red]")
        if platform.system() == "Darwin" and hardware_info["is_apple_silicon"] and "metal" in str(e).lower():
            console.print("[yellow]Hint: Metal errors? Reinstall with: CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install -U llama-cpp-python --no-cache-dir[/yellow]")
        elif hardware_info["cuda_available"] and ("cuda" in str(e).lower() or "cublas" in str(e).lower()):
             console.print("[yellow]Hint: CUDA errors? Check drivers or reinstall with: CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir[/yellow]")
        raise


# --- Main Processing Logic ---
def process_model(model_path, questions_df, config):
    """Process a single model on all questions"""
    model_name = os.path.basename(model_path)
    log_file = os.path.join(config["results_dir"], "benchmark_log.txt")
    hardware_info = detect_hardware() # Needed for cleanup check

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold]Model:[/bold] {model_name}\n"
            f"[bold]Valid Questions:[/bold] {len(questions_df)}\n"
            f"[bold]Timeout:[/bold] {config['timeout_seconds']}s/question",
            title=f"Starting: [magenta]{model_name}[/magenta]", border_style="green", expand=False
        ))
    else:
        console.print(f"--- Starting: {model_name} ({len(questions_df)} valid questions) ---")

    model_results = []
    model_instance = None
    original_handler = None

    try:
        model_instance = initialize_llama_model(model_path, config)

        if platform.system() != "Windows":
            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        # else: console.print("[yellow]Note: Timeout not enforced on Windows.[/yellow]") # Reduce verbosity

        total_questions_to_process = len(questions_df)
        processed_question_count = 0
        progress_context = None # Define progress context manager

        if RICH_AVAILABLE:
             progress_context = Progress(
                TextColumn("[progress.description]{task.description}"), BarColumn(bar_width=None),
                TaskProgressColumn(), TextColumn("Acc:{task.fields[accuracy]:.1%}"),
                TimeElapsedColumn(), TextColumn("ETA:"), TimeRemainingColumn(),
                console=console, transient=False # Keep finished bars visible
             )
             progress = progress_context.__enter__() # Manually enter context
             task = progress.add_task(f"[cyan]{model_name}[/cyan]", total=total_questions_to_process, accuracy=0.0)
        else: # Basic progress for non-rich consoles
             console.print(f"Processing {total_questions_to_process} questions for {model_name}...")


        for idx, row in questions_df.iterrows(): # Use valid_df index
            # --- Prepare ---
            question_text = str(row["Question"])
            choices_text = str(row["All Answer Choices"])
            correct_letter = row["correct_letter"] # Known to be valid
            prompt = build_universal_prompt(question_text, choices_text)
            processed_question_count += 1

            start_time = time.time(); model_raw_response = ""; model_answer_letter = None
            is_correct = False; timed_out = False; error_message = None

            # --- Run ---
            try:
                if platform.system() != "Windows": signal.alarm(config["timeout_seconds"])
                output = model_instance(
                    prompt=prompt, max_tokens=config["max_tokens"], temperature=config["temperature"],
                    top_p=config["top_p"], top_k=config["top_k"], echo=config["echo"],
                    stop=["`Final Answer:", "\n**Instruction**:", "\n**Question**:"]
                )
                if platform.system() != "Windows": signal.alarm(0)

                if output and output["choices"]:
                    model_raw_response = output["choices"][0].get("text", "").strip()
                    model_answer_letter = parse_final_answer_letter(model_raw_response)   

                is_correct = (model_answer_letter == correct_letter)
                inference_time = time.time() - start_time
            except TimeoutException:
                model_raw_response = "TIMEOUT"; inference_time = config["timeout_seconds"]
                timed_out = True; error_message = "Timeout"
            except KeyboardInterrupt: raise # Propagate interruption
            except Exception as e:
                console.print(f"\n[red]Inference Error Q#{row.name+2}: {e}[/red]")
                model_raw_response = f"ERROR: {e}"; inference_time = time.time() - start_time
                error_message = str(e)
            finally:
                if platform.system() != "Windows": signal.alarm(0)

            # --- Store & Log ---
            result = {
                "model": model_name, "question_id_csv": row.name,
                "question_text_preview": question_text[:100]+"...", "correct_letter": correct_letter,
                "model_raw_response": model_raw_response, "model_answer_letter": model_answer_letter,
                "is_correct": is_correct, "inference_time": inference_time,
                "timed_out": timed_out, "error": error_message
            }
            model_results.append(result)

            # --- Update Progress ---
            if RICH_AVAILABLE:
                current_correct = sum(r["is_correct"] for r in model_results)
                current_accuracy = current_correct / processed_question_count if processed_question_count > 0 else 0.0
                progress.update(task, advance=1, accuracy=current_accuracy)
            elif processed_question_count % 10 == 0: # Basic progress update every 10 Qs
                 console.print(f"... {processed_question_count}/{total_questions_to_process} done.")

            # --- Log to File ---
            with open(log_file, "a", encoding="utf-8") as f:
                 status = 'Correct' if is_correct else 'Incorrect'
                 if timed_out: status += ' (TIMEOUT)'
                 elif error_message: status += f' (ERROR)'
                 f.write(f"\n---\nModel: {model_name}\nQ# (CSV Line {row.name + 2})\n")
                 f.write(f"Correct: {correct_letter}\nParsed: {model_answer_letter}\nStatus: {status}\n")
                 f.write(f"Time: {inference_time:.2f}s\n")
                 # Optionally log more details

        # --- Cleanup Progress Bar ---
        if progress_context:
            progress_context.__exit__(None, None, None) # Manually exit context

    finally: # Ensure cleanup happens even if errors occur mid-processing
        if platform.system() != "Windows" and original_handler is not None:
            try: signal.signal(signal.SIGALRM, original_handler)
            except ValueError: pass
        if model_instance is not None:
            console.print("\n[cyan]Releasing model resources...[/cyan]")
            del model_instance; model_instance = None; gc.collect()
            if TORCH_AVAILABLE and config.get("use_gpu", False) and hardware_info['cuda_available']:
                 try: torch.cuda.empty_cache()
                 except Exception: pass # Ignore cache clearing errors
            # console.print("[green]Model resources released.[/green]") # Reduce verbosity

    # --- Analysis and Reporting ---
    if not model_results:
         console.print(f"[red]No results generated for {model_name}.[/red]")
         return None

    results_df = pd.DataFrame(model_results)
    output_csv = os.path.join(config["results_dir"], f"{model_name}_results_detailed.csv")
    try: results_df.to_csv(output_csv, index=False, encoding='utf-8')
    except Exception as e: console.print(f"[red]Error saving CSV for {model_name}: {e}[/red]")
    console.print(f"Detailed results saved to: {output_csv}")

    valid_results_df = results_df[~results_df['timed_out'] & results_df['error'].isna()]
    num_valid = len(valid_results_df); num_correct = valid_results_df["is_correct"].sum()
    accuracy = num_correct / num_valid if num_valid > 0 else 0.0
    avg_time = valid_results_df['inference_time'].mean() if num_valid > 0 else 0.0
    timeouts = results_df["timed_out"].sum(); errors = results_df['error'].notna().sum() - timeouts

    if RICH_AVAILABLE:
        summary_table = Table(title=f"Results: [magenta]{model_name}[/magenta]", show_header=False, box=rich.box.HEAVY_EDGE)
        summary_table.add_column("Metric", style="cyan", justify="right")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Questions Attempted:", str(len(results_df)))
        summary_table.add_row("Valid Responses:", str(num_valid))
        summary_table.add_row("Correct (on Valid):", f"{num_correct}")
        summary_table.add_row("Accuracy (on Valid):", f"{accuracy:.2%}")
        summary_table.add_row("Avg. Time (Valid):", f"{avg_time:.2f} s")
        summary_table.add_row("Timeouts:", f"[red]{timeouts}[/red]" if timeouts > 0 else "0")
        summary_table.add_row("Other Errors:", f"[red]{errors}[/red]" if errors > 0 else "0")
        console.print(summary_table)
    else: # Basic print
         console.print(f"\n--- Results: {model_name} ---")
         console.print(f"  Attempted: {len(results_df)}, Valid: {num_valid}, Correct: {num_correct}")
         console.print(f"  Accuracy (Valid): {accuracy:.2%}")
         console.print(f"  Avg Time (Valid): {avg_time:.2f} s")
         console.print(f"  Timeouts: {timeouts}, Errors: {errors}")

    console.rule() # Separator before next model

    return results_df


# --- Visualization ---
def create_visualizations(all_results_df, config):
    """Create and save visualizations of the benchmark results"""
    if all_results_df is None or all_results_df.empty:
        console.print("[yellow]No combined results for visualization.[/yellow]")
        return None

    # Calculate summary metrics per model
    summary_list = []
    for model_name in all_results_df['model'].unique():
        model_df = all_results_df[all_results_df['model'] == model_name]
        valid_df = model_df[~model_df['timed_out'] & model_df['error'].isna()]
        num_valid = len(valid_df); num_correct = valid_df['is_correct'].sum()
        accuracy = num_correct / num_valid if num_valid > 0 else 0.0
        avg_time = valid_df['inference_time'].mean() if num_valid > 0 else 0.0
        total_attempted = len(model_df)
        timeouts = model_df['timed_out'].sum()
        errors = model_df['error'].notna().sum() - timeouts
        summary_list.append({
            'model': model_name, 'accuracy': accuracy, 'avg_time': avg_time,
            'timeouts': timeouts, 'errors': errors, 'valid_responses': num_valid,
            'total_attempted': total_attempted
        })
    summary_df = pd.DataFrame(summary_list)
    if summary_df.empty: return None

    summary_path = os.path.join(config["results_dir"], "model_comparison_summary.csv")
    try: summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    except Exception as e: console.print(f"[red]Error saving summary CSV: {e}[/red]")
    console.print(f"Summary metrics saved to: {summary_path}")

    sorted_summary = summary_df.sort_values('accuracy', ascending=False).reset_index()
    num_models = len(sorted_summary)
    plt_success = True # Track if plots are generated

    # --- Accuracy Bar Chart ---
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(max(8, num_models * 1.0), 6))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_models))
        bars = ax.bar(sorted_summary.index, sorted_summary['accuracy'], color=colors, zorder=2, width=0.7)
        ax.set_title("Model Accuracy (on Valid Responses)", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=12); ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.5)
        ax.set_xticks(sorted_summary.index); ax.set_xticklabels(sorted_summary['model'], rotation=30, ha="right", fontsize=9)
        ax.tick_params(axis='x', pad=5)
        for bar in bars: # Add labels
            height = bar.get_height()
            ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        path = os.path.join(config["results_dir"], "accuracy_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
        console.print(f"✓ Accuracy chart saved.")
    except Exception as e:
        console.print(f"[red]Error creating accuracy chart: {e}[/red]"); plt_success = False

    # --- Accuracy vs. Time Scatter Plot ---
    try:
        fig, ax = plt.subplots(figsize=(11, 7))
        min_size, max_size = 50, 400; size_data = sorted_summary['valid_responses'].fillna(0)
        sizes = [(min_size + max_size)/2] * num_models
        if size_data.max() > size_data.min(): sizes = min_size + (max_size - min_size) * (size_data - size_data.min()) / (size_data.max() - size_data.min())
        scatter = ax.scatter(sorted_summary['avg_time'], sorted_summary['accuracy'], s=sizes,
                             c=sorted_summary['accuracy'], cmap='viridis_r', alpha=0.8, vmin=0, vmax=1, zorder=2, edgecolors='k', linewidth=0.5)
        for i, row in sorted_summary.iterrows(): # Add labels
             ax.text(row['avg_time']*1.01, row['accuracy'], f" {row['model']}", fontsize=8, va='center',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, ec="none"))
        cbar = fig.colorbar(scatter, ax=ax); cbar.set_label('Accuracy', rotation=270, labelpad=15)
        ax.set_title("Accuracy vs. Avg Inference Time", fontsize=14)
        ax.set_xlabel("Avg Time per Question (s, Valid Responses)", fontsize=12)
        ax.set_ylabel("Accuracy (Valid Responses)", fontsize=12)
        ax.set_ylim(-0.05, 1.05); ax.set_xlim(left=-max(1, sorted_summary['avg_time'].max())*0.05)
        ax.grid(True, linestyle=':', alpha=0.6); fig.tight_layout()
        path = os.path.join(config["results_dir"], "accuracy_vs_time.png")
        fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
        console.print(f"✓ Accuracy vs. time plot saved.")
    except Exception as e:
        console.print(f"[red]Error creating scatter plot: {e}[/red]"); plt_success = False

    if plt_success: console.print(f"[bold green]Visualizations generated in: {config['results_dir']}[/bold green]")
    return sorted_summary


# --- Main Execution ---
def main():
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    def handle_exit_signal(signum, frame):
        console.print("\n[bold yellow]Ctrl+C detected. Shutting down...[/bold yellow]")
        signal.signal(signal.SIGINT, original_sigint_handler)
        console.print("[yellow]Benchmark interrupted.[/yellow]")
        sys.exit(1)
    signal.signal(signal.SIGINT, handle_exit_signal)

    global LLAMA_VERBOSE
    LLAMA_VERBOSE = False # Default

    try:
        display_welcome_screen()
        config = get_user_input()

        log_file = os.path.join(config["results_dir"], "benchmark_log.txt")
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"LLM Neurosurgery Benchmark Log\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nConfig:\n")
                for k, v in config.items(): f.write(f"  {k}: {[os.path.basename(p) for p in v] if k == 'model_paths' else v}\n")
                f.write("---\n")
            console.print(f"Logging detailed progress to: {log_file}")
        except Exception as e: console.print(f"[red]Error setting up log file {log_file}: {e}[/red]"); return

        console.print(f"\nLoading questions from: [cyan]{config['csv_path']}[/cyan]...")
        try: df_full = pd.read_csv(config['csv_path'], dtype=str, keep_default_na=False)
        except Exception as e: console.print(f"[bold red]Fatal Error loading CSV: {e}[/bold red]"); return

        df = df_full.head(config['num_questions']).copy()
        def extract_correct_letter(answer_str):
            if not isinstance(answer_str, str) or not answer_str.strip(): return None
            match = re.match(r"^\s*([A-E])(?:[.\s:]|$)", answer_str.strip(), re.IGNORECASE)
            return match.group(1).upper() if match else None
        df["correct_letter"] = df["Correct Answer"].apply(extract_correct_letter)

        invalid_questions_df = df[df["correct_letter"].isna()]
        if not invalid_questions_df.empty:
            num_invalid = len(invalid_questions_df)
            console.print(f"\n[bold yellow]WARNING: {num_invalid}/{len(df)} questions have invalid 'Correct Answer'.[/bold yellow]")
            if RICH_AVAILABLE:
                invalid_table = Table(title=f"Invalid 'Correct Answer' Examples ({min(10, num_invalid)} shown)", box=rich.box.SIMPLE)
                invalid_table.add_column("CSV Line#", style="cyan", justify="right")
                invalid_table.add_column("Invalid String", style="red")
                for idx, row in invalid_questions_df.head(10).iterrows(): invalid_table.add_row(str(idx + 2), repr(row["Correct Answer"]))
                console.print(invalid_table)
            console.print(f"\n[yellow]These {num_invalid} questions will be SKIPPED. Continue? (y/n)[/yellow]")
            if input("> ").strip().lower() != 'y': console.print("[red]Aborted.[/red]"); return
            df_valid = df.dropna(subset=["correct_letter"]).copy()
            console.print(f"[green]Proceeding with {len(df_valid)} valid questions.[/green]\n")
        else:
            df_valid = df; console.print(f"[green]All {len(df_valid)} questions valid.[/green]\n")

        if df_valid.empty: console.print("[red]No valid questions remaining.[/red]"); return

        all_results_list = []; start_benchmark_time = time.time()
        for model_path in config["model_paths"]:
            if not os.path.exists(model_path):
                 console.print(f"[red]ERROR: Model not found: {model_path}. Skipping.[/red]")
                 with open(log_file, "a", encoding="utf-8") as f: f.write(f"\n---\nERROR: Model skipped (not found): {model_path}\n---\n")
                 continue
            console.rule(f"[bold blue]Processing: {os.path.basename(model_path)}[/bold blue]")
            try:
                model_results_df = process_model(model_path, df_valid, config)
                if model_results_df is not None and not model_results_df.empty: all_results_list.append(model_results_df)
            except KeyboardInterrupt: console.print("\n[yellow]Interrupted. Moving to final analysis...[/yellow]"); break
            except Exception as e:
                 console.print(f"[bold red]CRITICAL ERROR processing {os.path.basename(model_path)}: {e}[/bold red]")
                 with open(log_file, "a", encoding="utf-8") as f:
                      f.write(f"\n---\nCRITICAL ERROR: {os.path.basename(model_path)}\nError: {e}\nTraceback:\n"); import traceback; traceback.print_exc(file=f); f.write("---\n")

        end_benchmark_time = time.time(); total_duration = end_benchmark_time - start_benchmark_time
        console.print(f"\nTotal benchmark time: {total_duration // 60:.0f}m {total_duration % 60:.1f}s.")
        if all_results_list:
            console.print("\n[bold magenta]--- Final Analysis ---[/bold magenta]")
            combined_results = pd.concat(all_results_list, ignore_index=True)
            
            # Ask if user wants to manually review incorrect answers
            console.print("\n[bold cyan]Would you like to manually review incorrect answers? (y/n)[/bold cyan]")
            if input("> ").strip().lower() == 'y':
                # Add a review_note column for tracking changes
                combined_results['review_note'] = ""
                
                # Run manual review
                combined_results = manual_review(combined_results, df_valid)
                
                # Save reviewed results
                reviewed_path = os.path.join(config["results_dir"], "all_models_results_reviewed.csv")
                try: 
                    combined_results.to_csv(reviewed_path, index=False, encoding='utf-8')
                    console.print(f"[green]Reviewed results saved: {reviewed_path}[/green]")
                except Exception as e: 
                    console.print(f"[red]Error saving reviewed CSV: {e}[/red]")
            
            combined_path = os.path.join(config["results_dir"], "all_models_results_detailed.csv")
            try: combined_results.to_csv(combined_path, index=False, encoding='utf-8'); console.print(f"Combined results saved: {combined_path}")
            except Exception as e: console.print(f"[red]Error saving combined CSV: {e}[/red]")

            summary_df = create_visualizations(combined_results, config)

            if summary_df is not None and not summary_df.empty and RICH_AVAILABLE:
                console.print("\n[bold underline]Overall Model Performance:[/bold underline]")
                summary_table = Table(title="Model Summary (Accuracy on Valid Responses)", show_header=True, header_style="bold blue", box=rich.box.DOUBLE_EDGE)
                summary_table.add_column("Model", style="cyan", min_width=20, no_wrap=True); summary_table.add_column("Accuracy", style="bold green", justify="right")
                summary_table.add_column("Avg Time(s)", style="yellow", justify="right"); summary_table.add_column("Valid/Attempted", style="white", justify="center")
                summary_table.add_column("Timeouts", style="red", justify="right"); summary_table.add_column("Errors", style="red", justify="right")
                for _, row in summary_df.iterrows():
                     summary_table.add_row(row['model'], f"{row['accuracy']:.1%}", f"{row['avg_time']:.2f}",
                                          f"{int(row['valid_responses'])}/{int(row['total_attempted'])}", str(int(row['timeouts'])), str(int(row['errors'])))
                console.print(summary_table)
            elif summary_df is not None and not summary_df.empty: # Basic print
                 console.print("\n--- Overall Summary ---")
                 console.print(summary_df.to_string(index=False, formatters={'accuracy': '{:.1%}'.format, 'avg_time': '{:.2f}'.format}))

            console.print(f"\n[bold green]Benchmark finished! Results: {config['results_dir']}[/bold green]")
            console.print("\nOpen results directory? (y/n)")
            if input("> ").strip().lower() in ('y', 'yes'):
                try:
                    rp = os.path.abspath(config["results_dir"]); console.print(f"Opening: {rp}")
                    if platform.system()=="Windows": os.startfile(rp)
                    elif platform.system()=="Darwin": subprocess.run(["open", rp], check=True)
                    else: subprocess.run(["xdg-open", rp], check=True)
                except Exception as e: console.print(f"[red]Could not open directory: {e}[/red]")
        else:
            console.print("\n[bold red]Benchmark completed, but no results generated.[/bold red]")
            console.print(f"Check logs: {log_file}")

    except KeyboardInterrupt: console.print("\n[yellow]Benchmark aborted during setup.[/yellow]")
    except Exception as e:
         console.print(f"\n[bold red]Critical Error: {e}[/bold red]"); import traceback; traceback.print_exc()
    finally:
         signal.signal(signal.SIGINT, original_sigint_handler)
         console.print("\nExiting.")


if __name__ == "__main__":
    # --- Requirements Check ---
    missing_deps = []
    dependency_info = [
        ("rich", "rich", True), # Module, Package, Required?
        ("pandas", "pandas", True),
        ("numpy", "numpy", True),
        ("matplotlib", "matplotlib", True),
        ("llama_cpp", "llama-cpp-python", True)
    ]
    print("Checking dependencies...")
    for module_name, package_name, is_required in dependency_info:
        try: __import__(module_name); print(f"  [green]✓ {package_name}[/green]")
        except ImportError:
            if is_required: missing_deps.append(package_name)
            print(f"  [red]✗ {package_name} (Missing){' *Required*' if is_required else ''}[/red]")

    # Check torch separately
    if not TORCH_AVAILABLE: print("  [yellow]! torch (Optional, needed for CUDA GPU)[/yellow]")
    else: print("  [green]✓ torch[/green]")

    if missing_deps:
        # Use console if available for error message
        console.print(f"\n[bold red]Error: Missing required packages: {', '.join(missing_deps)}[/bold red]")
        console.print("Install using pip:"); console.print(f"  [bold]pip install {' '.join(missing_deps)}[/bold]")
        if not TORCH_AVAILABLE:
             console.print("\n[yellow]Note: PyTorch ('torch') recommended for NVIDIA GPU support.[/yellow]")
             console.print("[yellow]Install from https://pytorch.org/get-started/locally/[/yellow]")
        sys.exit(1)
    else: console.print("[bold green]All required dependencies found.[/bold green]\n")

    # --- Run Main Application ---
    main()