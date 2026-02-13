# 🎵 Taylor Swift Lyrics Generator

AI-powered lyrics generation using a GPT-2 transformer trained on Taylor Swift's discography.

**[Try Live Demo →](https://huggingface.co/spaces/chz011/taylor-swift-lyrics-generator)**

## Features

- Generate lyrics in different Taylor Swift album styles
- 29M parameter transformer model
- Interactive Gradio web interface
- Customizable creativity settings

## Getting the Model

The trained model (336MB) is hosted on HuggingFace Spaces due to GitHub's size limits.

**To run locally:**
1. Download the model `best_model.pt` from [HuggingFace Spaces](https://huggingface.co/spaces/chz011/taylor-swift-lyrics-generator)
2. Place `best_model.pt` in the project root
3. Run the following code:

```bash
git clone https://github.com/YOUR_USERNAME/AI_TSlyrics_generator.git
cd AI_TSlyrics_generator
pip install -r requirements.txt
python app.py
```

4. Open `http://localhost:7860`

## Tech Stack

PyTorch • Transformers • Gradio • Python 3.10+

## Model Performance
- **Validation Loss**: 5.46
- **Perplexity**: 239.81
- **Parameters**: 29.01M

**Chi Zhang** | [LinkedIn](https://www.linkedin.com/in/chi-zhang-6904a3257/) | [Portfolio](link)