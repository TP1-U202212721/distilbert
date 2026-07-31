import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Los pesos no viajan en el repo (.gitignore excluye los .safetensors), asi
# que se cargan desde el Hub. El dockerfile los precarga durante el build,
# por lo que en runtime esto resuelve contra el cache local sin bajar nada.
MODEL_PATH = "LordMathi2741/distilbert-es"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def test_inference(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return torch.argmax(outputs.logits, dim=1).item()