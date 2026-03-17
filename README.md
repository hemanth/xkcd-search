# XKCD Reverse Lookup 
> Upload an XKCD comic image or describe it in text — instantly find which comic it is.

Powered by **gemini-embedding-2-preview** multimodal embeddings and the [olivierdehaene/xkcd](https://huggingface.co/datasets/olivierdehaene/xkcd) dataset.

![Architecture](arch.png)


https://github.com/user-attachments/assets/9ed0f3d9-eecd-4257-88be-3e3e3f0359be

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
cp .env.example .env
# edit .env with your key

# Fetch comics from HF dataset (default: last 50)
python fetch_xkcd.py 50

# Build the embedding index
python index_comics.py

# Start the server
python app.py
```

Open [http://localhost:8000](http://localhost:8000) and search!

## How It Works

1. **Fetch** — Loads XKCD metadata from the HF dataset and downloads comic images.
2. **Index** — Embeds each comic image using `gemini-embedding-2-preview` into a shared vector space.
3. **Search** — Uploaded images *or* text queries are embedded with the same model. Cosine similarity ranks the closest comics.

Because text and images share the same embedding space, you can describe a comic ("someone flying with a python script") and find it just as well as uploading a screenshot.
