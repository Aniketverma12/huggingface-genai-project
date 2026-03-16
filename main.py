from transformers import pipeline
from PIL import Image
import requests


def sentiment_demo():
    print("\n--- Sentiment Analysis ---")
    sentiment = pipeline("sentiment-analysis")
    text = "This course on Generative AI is very useful."
    result = sentiment(text)
    print("Input:", text)
    print("Output:", result)


def text_generation_demo():
    print("\n--- Text Generation ---")
    generator = pipeline("text-generation")
    prompt = "The future of artificial intelligence is"
    result = generator(prompt, max_new_tokens=30, do_sample=True)
    print(result[0]["generated_text"])


def question_answering_demo():
    print("\n--- Question Answering ---")
    qa = pipeline("question-answering")
    context = (
        "Hugging Face is a company that develops tools for machine learning "
        "and natural language processing."
    )
    question = "What does Hugging Face develop?"
    result = qa(question=question, context=context)
    print("Question:", question)
    print("Answer:", result["answer"])


def translation_demo():
    print("\n--- Translation (EN -> HI) ---")
    translator = pipeline("translation_en_to_fr")  # safe default in transformers
    text = "Machine learning makes systems smarter."
    result = translator(text)
    print("Input:", text)
    print("Output:", result)


def image_classification_demo():
    print("\n--- Image Classification ---")
    classifier = pipeline("image-classification")
    url = "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    result = classifier(image)
    print(result[:3])


if __name__ == "__main__":
    sentiment_demo()
    text_generation_demo()
    question_answering_demo()
    translation_demo()
    image_classification_demo()
