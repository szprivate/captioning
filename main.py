import argparse
import ollama
import gradio as gr
from pathlib import Path
from typing import Generator

# A set of common image file extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def create_description(
    client: ollama.Client, image_path: Path, output_path: Path, model: str, prompt: str, activation_text: str
) -> Generator[str, None, None]:
    """
    Generates a description for a single image using Ollama and saves it.

    Args:
        client: The Ollama client instance.
        image_path: The path to the input image file.
        output_path: The path where the output text file will be saved.
        model: The name of the Ollama model to use.
        prompt: The prompt to send to the model along with the image.
        activation_text: Text to prepend to the description.

    Returns:
        A generator of log strings.
    """
    yield f"Processing {image_path.name}..."
    try:
        # Ensure the image path is a string for the API call
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)],
                }
            ],
        )
        description = response["message"]["content"]

        # Prepend activation text if it exists
        final_description = (
            f"{activation_text} {description}" if activation_text else description
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_description)
        yield f"  -> Saved description to {output_path.name}"

    except ollama.ResponseError as e:
        yield f"  -> Error processing {image_path.name}: {e.error}"
        if "model not found" in e.error:
            yield (
                f"  -> Please make sure you have pulled the model with 'ollama pull {model}'"
            )
    except Exception as e:
        yield f"  -> An unexpected error occurred with {image_path.name}: {e}"


def process_images_in_folder(
    client: ollama.Client, input_dir: Path, output_dir: Path, model: str, prompt: str, activation_text: str, force: bool
) -> Generator[str, None, None]:
    """
    Iterates through images in a folder, generates descriptions, and saves them.

    Returns:
        A generator of log strings from the process.
    """
    if not input_dir.is_dir():
        yield f"Error: Input directory not found at '{input_dir}'"
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    yield f"Reading images from: {input_dir.resolve()}"
    yield f"Saving descriptions to: {output_dir.resolve()}"
    yield f"Using model: {model}"
    yield "-" * 20

    image_files = [
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        yield "No supported image files found in the input directory."
        return

    for image_path in image_files:
        output_filename = image_path.stem + ".txt"
        output_path = output_dir / output_filename

        if not force and output_path.exists():
            yield f"Skipping {image_path.name}, description already exists."
            continue

        yield from create_description(client, image_path, output_path, model, prompt, activation_text)


def launch_gradio_app():
    """Launches the Gradio web UI for the image description tool."""

    def gradio_process_wrapper(input_dir, output_dir, host, model, prompt, activation_text, force):
        if not input_dir:
            return "Error: Input directory must be provided."

        # If output_dir is not provided, use input_dir
        if not output_dir:
            output_dir = input_dir

        try:
            client = ollama.Client(host=host)
            # A simple check to see if the server is reachable
            client.list()
        except Exception as e:
            return f"Could not connect to Ollama at {host}. Please ensure it's running.\n\nError: {e}"

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        log_generator = process_images_in_folder(
            client=client,
            input_dir=input_path,
            output_dir=output_path,
            model=model,
            prompt=prompt,
            activation_text=activation_text,
            force=force
        )
        return "\n".join(log_generator)

    def update_model_list(host_url: str):
        """Queries the Ollama server and updates the model dropdown."""
        try:
            print(f"Attempting to connect to Ollama at {host_url} to fetch models...")
            client = ollama.Client(host=host_url)
            models_info = client.list()
            # The response is a dictionary with a 'models' key containing a list of models.
            model_names = sorted([model['model'] for model in models_info['models']])
            print(f"Successfully fetched {len(model_names)} models.")

            preferred_default_model = "qwen3-vl:30b"
            selected_model = None
            if preferred_default_model in model_names:
                selected_model = preferred_default_model
            elif model_names:
                selected_model = model_names[0]

            return gr.Dropdown(choices=model_names, value=selected_model, interactive=True)
        except Exception as e:
            print(f"Warning: Could not connect to Ollama to get model list: {e}")
            # Return an empty, but still interactive, dropdown
            return gr.Dropdown(choices=[], value=None, interactive=True)

    # Get default values from a dummy parser to avoid hardcoding
    dummy_parser = argparse.ArgumentParser()
    dummy_parser.add_argument("--host", default="http://localhost:11434")
    dummy_parser.add_argument("--prompt", default="Describe this image in detail.")
    defaults = dummy_parser.parse_args([])

    with gr.Blocks(title="sz captioning tool") as demo:
        gr.Markdown("# sz captioning tool")
        gr.Markdown(
            "A tool to generate text descriptions for all images in a folder. Fill in the details and click 'Submit'.")

        with gr.Row():
            input_dir_box = gr.Textbox(label="Input Directory", placeholder="/path/to/your/images")
            output_dir_box = gr.Textbox(label="Output Directory (optional)", placeholder="Defaults to input directory")

        with gr.Row(equal_height=False):
            host_box = gr.Textbox(label="Ollama Host", value=defaults.host, container=True)
            model_dropdown = gr.Dropdown(label="Model", choices=[], allow_custom_value=True, interactive=True, container=True)
            refresh_button = gr.Button("Refresh Models", size="sm")

        prompt_box = gr.Textbox(label="Prompt", value=defaults.prompt)
        activation_text_box = gr.Textbox(label="Activation Text", placeholder="e.g., a photo of")
        force_checkbox = gr.Checkbox(label="Force Regeneration", value=False)

        submit_button = gr.Button("Submit", variant="primary")
        log_output = gr.Textbox(label="Log", lines=20, interactive=False)

        # --- Event Handlers ---
        # Chain button clicks and app load to the model update function
        refresh_button.click(fn=update_model_list, inputs=[host_box], outputs=[model_dropdown])
        demo.load(fn=update_model_list, inputs=[host_box], outputs=[model_dropdown])

        # Main process trigger
        submit_button.click(
            fn=gradio_process_wrapper,
            inputs=[input_dir_box, output_dir_box, host_box, model_dropdown, prompt_box, activation_text_box, force_checkbox],
            outputs=log_output
        )

    demo.launch(server_name="0.0.0.0")


def main():
    """
    Main function to parse arguments and start the image processing.
    """
    parser = argparse.ArgumentParser(
        description="Generate descriptions for images in a folder using Ollama.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=None,
        help="Path to the folder containing images. Required for CLI mode.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Path to the folder where description files will be saved. \n(default: same as input directory)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server host and port. \n(default: http://localhost:11434)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3-vl:30b",
        help="The multimodal model to use (e.g., 'llava', 'qwen3-vl:30b'). \n(default: qwen3-vl:30b)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="The prompt to use for generating the description. \n(default: 'Describe this image in detail.')",
    )
    parser.add_argument(
        "-at",
        "--activation-text",
        type=str,
        default="",
        help="Text to prepend to every generated description (e.g., 'a photo of').",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force regeneration of descriptions even if they already exist.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Gradio web UI instead of running in the console.",
    )

    args = parser.parse_args()

    if args.gui:
        print("Launching Gradio UI...")
        print("Access it in your browser at http://localhost:7860 or http://<your-ip-address>:7860")
        launch_gradio_app()
        return

    # For CLI mode, input_dir is required.
    if not args.input_dir:
        parser.error("argument -i/--input-dir is required for CLI mode.")

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir) if args.output_dir else input_path

    # --- CLI Execution ---
    client = ollama.Client(host=args.host)
    log_generator = process_images_in_folder(
        client=client,
        input_dir=input_path,
        output_dir=output_path,
        model=args.model,
        prompt=args.prompt,
        activation_text=args.activation_text,
        force=args.force
    )
    for log_line in log_generator:
        print(log_line)


if __name__ == "__main__":
    main()