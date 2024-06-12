from ollama import OllamaModel

# Load the Llama3:70b model
model_name = "llama3:70b"
model = OllamaModel(model_name)

# Check model information
print(model)


