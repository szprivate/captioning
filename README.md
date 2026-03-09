SIMPLE BATCH CAPTIONING TOOL

This small tool will connect to present ollama servers, fetch a list of the available ollama models.
It comes with a small gradio UI, you can set input / output folder, prompt for how the captions should be created, and a field for an activation string.
Activation string will be added upfront.

Recommended model to use: qwen3-vl:30b

To use this, you'll need to install ollama beforehand:
https://ollama.com/download/windows

After installation, fetch your ollama model with cli command:
```ollama pull [model_name]```

Then, run ollama server:
```ollama serve```

create a venv, activate it, then run the tool via:
```python ./main --gui```

Access gradio app here:
http://localhost:7860/
