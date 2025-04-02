import os
import argparse
import questionary # For interactive prompts
from rich.console import Console # For better printing
from rich.panel import Panel
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.table import Table
import sys # To exit gracefully
import statistics # For calculating average scores

# Attempt to import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    print("\nError: llama-cpp-python is not installed.")
    print("Please install it using: pip install llama-cpp-python")
    print("For GPU support, refer to the llama-cpp-python documentation.\n")
    sys.exit(1) # Exit if library not found


# --- Configuration ---
console = Console()
DEFAULT_N_CTX = 4048 # Default context window size (adjust if needed)
DEFAULT_MAX_TOKENS = 4768 # Increased default max tokens for potentially structured output
DEFAULT_N_GPU_LAYERS = 0 # Default GPU layers (0 = CPU only)


# --- Helper Functions ---

def find_gguf_models(model_dir):
    # (Same as previous version)
    if not os.path.isdir(model_dir):
        console.print(f"[bold red]Error:[/bold red] Directory not found: {model_dir}")
        return None
    gguf_files = []
    try:
        for filename in os.listdir(model_dir):
            if filename.lower().endswith(".gguf"):
                gguf_files.append(filename)
    except OSError as e:
        console.print(f"[bold red]Error reading directory {model_dir}:[/bold red] {e}")
        return None
    if not gguf_files:
        console.print(f"[yellow]No .gguf model files found in '{model_dir}'.[/yellow]")
        return None
    return sorted(gguf_files)


def load_local_model(model_path, n_ctx, n_gpu_layers, verbose=False):
    # (Same as previous version, added explicit console print on success)
    spinner = Spinner("dots", text=f" Loading model {os.path.basename(model_path)}...")
    llm = None
    with console.status(spinner):
        try:
            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose
            )
            # Explicit success message after loading
            console.print(f"[green]Model loaded successfully:[/green] {os.path.basename(model_path)}")
        except Exception as e:
            console.print(f"\n[bold red]Error loading model {os.path.basename(model_path)}:[/bold red]")
            console.print(f"{e}")
            console.print("[yellow]Check model path, file integrity, and system resources (RAM/VRAM).[/yellow]")
            llm = None
    return llm


def generate_local_completion(llm, prompt, max_tokens):
    # (Same as previous version)
    if llm is None:
        return "[Error: Model not loaded]", True
    response_text = None
    error_message = None
    spinner = Spinner("dots", text=" Generating response...")
    with console.status(spinner):
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                stop=None # Example stop: ["\nHuman:", "<|endoftext|>"]
            )
            # Check structure before accessing
            if response and 'choices' in response and len(response['choices']) > 0 and \
               'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                response_text = response['choices'][0]['message']['content'].strip()
                if not response_text:
                    error_message = "Model returned an empty response."
            else:
                 error_message = "Unexpected response structure from model."
                 console.print(f"\n[yellow]Debug: Raw response:[/yellow] {response}")


        except Exception as e:
            error_message = f"Error during generation: {e}"

    # Return the error message itself as the text if an error occurred
    return (error_message if error_message else response_text), bool(error_message)


def get_user_rating():
    """Prompts the user for a 1-5 rating."""
    rating = questionary.text(
        "Please rank the model's response quality (1=Low, 3=Medium, 5=Max/Human-level):",
        validate=lambda text: text.isdigit() and 1 <= int(text) <= 5 or "Please enter a number between 1 and 5."
    ).ask()
    if rating is None:
        return None # User cancelled
    return int(rating)

def display_summary(results_data):
    """Displays a summary table of average scores."""
    if not results_data:
        console.print("\n[yellow]No results recorded to summarize.[/yellow]")
        return

    console.rule("[bold]Demo Summary - Average Subjective Scores[/bold]")

    # Calculate average scores per model
    model_scores = {}
    for res in results_data:
        model = res['model']
        score = res['score']
        if model not in model_scores:
            model_scores[model] = {'total_score': 0, 'count': 0}
        model_scores[model]['total_score'] += score
        model_scores[model]['count'] += 1

    # Prepare table data
    table = Table(title="Average Model Quality Score (1-5)")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Average Score", style="magenta")
    table.add_column("Demos Rated", style="green")

    sorted_models = sorted(model_scores.keys())

    for model in sorted_models:
        data = model_scores[model]
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        table.add_row(model, f"{avg_score:.2f}", str(data['count']))

    console.print(table)


# --- Demo Specific Prompts ---

DEMO1_DATABASE_PROMPT_TEMPLATE = """
You are an AI assistant specializing in extracting structured clinical data from unstructured medical notes for a neurosurgical research database.
Analyze the provided clinical note(s) below. Extract the following specific pieces of information:

*   Patient MRN: (Medical Record Number, if available)
*   Date of Surgery: (YYYY-MM-DD format, if mentioned)
*   Procedure Name: (Specific name of the surgical procedure)
*   Tumor Size (mm): (Numerical value if mentioned, specify dimension if available e.g., APxWxH)
*   Tumor Location: (e.g., Left Frontal Lobe, Sellar, CP Angle)
*   EBL (cc): (Estimated Blood Loss in cc or mL)
*   Length of Stay (days): (If mentioned or calculable from dates)
*   Complications: (List any mentioned intra-operative or immediate post-operative complications, otherwise state 'None Reported')

Present the extracted data in a clear key-value list format below.
If a specific piece of information cannot be found in the provided text, clearly state 'Not Found'.

Clinical Note(s):
---
{user_notes}
---

Extracted Data:
"""

DEMO2_GCS_PROMPT_TEMPLATE = """
You are an AI assistant helping document the Glasgow Coma Scale (GCS) based on a clinical description.
The GCS components are:
*   Eye Opening (E): 4=Spontaneous, 3=To speech, 2=To pain, 1=None
*   Verbal Response (V): 5=Oriented, 4=Confused, 3=Inappropriate words, 2=Incomprehensible sounds, 1=None
*   Motor Response (M): 6=Obeys commands, 5=Localizes pain, 4=Withdraws from pain, 3=Flexion to pain (decorticate), 2=Extension to pain (decerebrate), 1=None

Analyze the following clinical description of a patient's status:
---
{user_description}
---

Perform the following steps:
1.  Determine the score for each component (E, V, M) based *only* on the description provided.
2.  Calculate the total GCS score.
3.  Format the result clearly listing E, V, M scores and the Total GCS (e.g., GCS: 15 (E4 V5 M6)).
4.  Draft 1-3 concise sentences suitable for a clinical note documenting this GCS assessment and the findings that support it.

Formatted GCS and Documentation:
"""


# --- New Demo Functions ---

def run_demo_database(llm, model_name, max_tokens):
    """Demo 1: Database Aggregator"""
    console.print(Panel("[bold cyan]Demo 1: Private AI Retrospective/Prospective Database Helper[/bold cyan]\nThis demo takes one or more clinical notes and attempts to extract predefined data fields suitable for building a research database. The quality depends heavily on the note's structure and the model's ability.", title="Demo Description", border_style="cyan"))
    console.print(f"You have selected [bold]{model_name}[/bold] to demo.")
    # Show prompt template - replace placeholder for clarity
    prompt_display = DEMO1_DATABASE_PROMPT_TEMPLATE.replace("{user_notes}", "[Your pasted notes will go here]")
    console.print(Panel(prompt_display, title="Model Prompt Template", border_style="dim", expand=False)) # expand=False keeps it collapsed initially
    console.print("You will provide your note or set of notes and evaluate the model's response.")

    user_notes = questionary.text(
        "Paste the clinical note(s) below:\nPress Enter twice (or Esc then Enter) to finish:",
        multiline=True,
        validate=lambda text: True if len(text.strip()) > 10 else "Please paste some note content."
    ).ask()

    if user_notes is None: return None # User cancelled

    final_prompt = DEMO1_DATABASE_PROMPT_TEMPLATE.format(user_notes=user_notes)

    console.print("\n[bold blue]Generating response locally...[/bold blue]")
    response_text, error = generate_local_completion(llm, final_prompt, max_tokens)

    console.print(Panel("Model's Raw Response", title=f"Output from {model_name}", border_style="blue"))
    if error:
        console.print(f"[bold red]Error or Empty Response:[/bold red]\n{response_text}")
    else:
        console.print(response_text) # Display raw extracted data format
    console.print("-" * 20 + " End Output " + "-" * 20)

    # Get Rating
    if not error: # Only ask for rating if there wasn't an explicit error
      rating = get_user_rating()
      return rating
    else:
      console.print("[yellow]Skipping rating due to generation error.[/yellow]")
      return None


def run_demo_gcs(llm, model_name, max_tokens):
    """Demo 2: Guideline Grader (GCS)"""
    console.print(Panel("[bold cyan]Demo 2: Clinical Criteria Documentation Helper (GCS Example)[/bold cyan]\nThis demo simplifies clinical criteria application using GCS. It demonstrates how local models might assist in scoring and documenting criteria based on unstructured input, simulating a quick note entry. **Requires careful clinical review.**", title="Demo Description", border_style="cyan"))
    console.print(f"You have selected [bold]{model_name}[/bold] to demo.")
    # Show prompt template
    prompt_display = DEMO2_GCS_PROMPT_TEMPLATE.replace("{user_description}", "[Your pasted description will go here]")
    console.print(Panel(prompt_display, title="Model Prompt Template", border_style="dim", expand=False))
    console.print("You will provide a clinical description (structured or unstructured) and evaluate the model's GCS assessment and documentation draft.")

    user_description = questionary.text(
        "Paste the clinical description of the patient's status below:\nPress Enter twice (or Esc then Enter) to finish:",
        multiline=True,
        validate=lambda text: True if len(text.strip()) > 5 else "Please describe the patient's status."
    ).ask()

    if user_description is None: return None # User cancelled

    final_prompt = DEMO2_GCS_PROMPT_TEMPLATE.format(user_description=user_description)

    console.print("\n[bold blue]Generating response locally...[/bold blue]")
    response_text, error = generate_local_completion(llm, final_prompt, max_tokens)

    console.print(Panel("Model's Raw Response", title=f"Output from {model_name}", border_style="blue"))
    if error:
        console.print(f"[bold red]Error or Empty Response:[/bold red]\n{response_text}")
    else:
        console.print(response_text) # Show GCS breakdown and draft text
    console.print("-" * 20 + " End Output " + "-" * 20)

    # Get Rating
    if not error:
      rating = get_user_rating()
      return rating
    else:
      console.print("[yellow]Skipping rating due to generation error.[/yellow]")
      return None


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI tool to demo local GGUF models on Neurosurgery tasks V2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # (Arguments same as previous version)
    parser.add_argument("--model-dir", required=False)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    parser.add_argument("--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    console.print(Panel("[bold magenta]Interactive Neurosurgery Demo CLI V2 (Local GGUF Models)[/bold magenta]", title="Welcome", border_style="magenta"))

    # --- Get Model Directory ---
    model_dir = args.model_dir
    if not model_dir:
        model_dir = questionary.path(
            "Enter the path to the directory containing your GGUF models:",
            only_directories=True,
            validate=lambda path: os.path.isdir(path) or "Path must be a valid directory."
        ).ask()
        if not model_dir:
            console.print("[bold red]Model directory path is required. Exiting.[/bold red]")
            return

    # --- Find Models ---
    available_models = find_gguf_models(model_dir)
    if not available_models:
        console.print("[bold red]Exiting as no models were found.[/bold red]")
        return

    # --- Main Interaction Loop ---
    loaded_llm = None
    selected_model_filename = None
    results_data = [] # List to store {'model': name, 'demo': demo_name, 'score': rating}

    while True:
        console.rule("[bold]Main Menu[/bold]")

        # --- Model Selection ---
        if not loaded_llm:
            console.print(f"Models found in: [cyan]{os.path.abspath(model_dir)}[/cyan]")
            model_choice = questionary.select(
                "Choose a GGUF model to load (use arrow keys):",
                choices=available_models + [questionary.Separator(), "Exit Program"],
                use_shortcuts=False
            ).ask()

            if model_choice is None or model_choice == "Exit Program":
                break
            selected_model_filename = model_choice
            full_model_path = os.path.join(model_dir, selected_model_filename)

            # --- Load Model ---
            loaded_llm = load_local_model(full_model_path, args.n_ctx, args.n_gpu_layers, args.verbose)
            if not loaded_llm:
                questionary.press_any_key_to_continue("Press any key to return to model selection...").ask()
                continue # Loading failed, loop back
        else:
            console.print(f"Current Model Loaded: [bold green]{selected_model_filename}[/bold green]")


        # --- Demo Selection ---
        demo_choice = questionary.select(
            "Which demo would you like to run?",
            choices=[
                "Demo 1: Database Helper",
                "Demo 2: Guideline Grader (GCS)",
                questionary.Separator(),
                "Change Loaded Model",
                "Show Score Summary",
                "Exit Program",
            ],
            use_shortcuts=True
        ).ask()

        rating = None # Reset rating for each demo run
        current_demo_name = ""

        if demo_choice is None or demo_choice == "Exit Program":
            break
        elif demo_choice == "Change Loaded Model":
            loaded_llm = None
            selected_model_filename = None
            console.print("\n[yellow]Model unloaded. Returning to model selection.[/yellow]\n")
            continue
        elif demo_choice == "Show Score Summary":
            display_summary(results_data)
            questionary.press_any_key_to_continue().ask()
            continue # Go back to menu after showing summary
        elif demo_choice == "Demo 1: Database Helper":
             current_demo_name = "Database Helper"
             rating = run_demo_database(loaded_llm, selected_model_filename, args.max_tokens)
        elif demo_choice == "Demo 2: Guideline Grader (GCS)":
             current_demo_name = "GCS Grader"
             rating = run_demo_gcs(loaded_llm, selected_model_filename, args.max_tokens)

        # Store result if rating was given
        if rating is not None:
            results_data.append({
                'model': selected_model_filename,
                'demo': current_demo_name,
                'score': rating
            })
            console.print(f"[green]Score of {rating} recorded for {selected_model_filename} on {current_demo_name}.[/green]")

        # Pause after demo before showing menu again
        if loaded_llm and demo_choice not in ["Change Loaded Model", "Show Score Summary"]:
             questionary.press_any_key_to_continue().ask()


    # --- Final Summary on Exit ---
    display_summary(results_data)
    console.print("\n[bold magenta]Exiting Neuro Demo CLI V2. Goodbye![/bold magenta]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user. Exiting.[/yellow]")
    except Exception as e:
         console.print(f"\n[bold red]An unexpected critical error occurred:[/bold red] {e}")
         import traceback
         traceback.print_exc()