## Ollama Setup

* Download and install ollama and follow GitHub.
* Run desktop app else follow the below.
* `ollama serve` starts the Ollama server if the desktop app is not running.
* `ollama run gemma3` runs a model, Downloads the model if not exist.
* `/bye` exits the LLM prompt.
* `ollama list` will list all installed models.

## Custom Ollama model with instructions

* Create `Modelfile` file with instructions.
* `ollama create <model_name> -f .\Modelfile` to create a custom model.
* `ollama run <model_name>` runs a custom model.