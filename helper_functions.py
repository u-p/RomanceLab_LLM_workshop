import matplotlib.pyplot as plt
import pandas

from huggingface_hub import HfApi
from nltk.tokenize import TweetTokenizer
from transformers import AutoConfig


word_tokenizer = TweetTokenizer().tokenize


def initialize_bos(model_name: str) -> bool:
    """
    Determines if the model requires a Beginning of Sentence (BOS) token based on its name.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if BOS is required, False otherwise.
    """
    if "gpt2" in model_name or "pythia" in model_name or "SmolLM" in model_name:
        return True
    return False


def plot_word_by_word_surprisals(surprisals: list, ymax=None, bar_col=None, title=None):
    """
    Plots a bar chart of word-by-word surprisal values.

    Args:
        surprisals (list): A list containing tuples of (token, surprisal_value).
        ymax (float, optional): The maximum value for the y-axis. If None, set to max surprisal value + 2.
        bar_col (str, optional): Color code for the bars. Defaults to "#1f77b4".
        title (str, optional): Title for the plot. Defaults to "Surprisal for each input".

    Notes:
        - Duplicate word labels are adjusted by appending an index (e.g., "word_1").
        - Surprisal values are displayed above each bar.
        - The x-axis labels are rotated for better readability.
        - The function displays the plot using matplotlib.
    """
    words, surprisal_vals = zip(*surprisals[0])

    # Adjust duplicate word labels
    seen = {}
    adjusted_words = []
    for word in words:
        if word in seen:
            seen[word] += 1
            adjusted_words.append(f"{word}_{seen[word]}")
        else:
            seen[word] = 0
            adjusted_words.append(word)

    if ymax is None:
        ymax = max(surprisal_vals) + 2

    if bar_col is None:
        bar_col = "#1f77b4"

    if title is None:
        title = "Surprisal for each input"

    plt.figure(figsize=(8, 4))
    bars = plt.bar(adjusted_words, surprisal_vals, color=bar_col, width=0.9)

    for bar, val in zip(bars, surprisal_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}',
                 ha='center', va='bottom', fontsize=9)

    plt.xlabel('')
    plt.ylabel('')
    plt.title(title)
    plt.xticks(rotation=85)
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.show()


def get_model_architecture_type(model_name: str) -> str:
    """
    Determines if a Hugging Face model is an encoder, decoder, or encoder-decoder.

    Args:
        model_name (str): The model ID on the Hugging Face Hub (e.g., "bert-base-uncased", "gpt2", "t5-small").

    Returns:
        str: "encoder-only", "decoder-only", "encoder-decoder", or "unknown".
    """
    try:
        config = AutoConfig.from_pretrained(model_name)

        # Encoder-decoder models typically have both encoder and decoder configurations.
        # They often have attributes like 'is_encoder_decoder' or 'encoder' and 'decoder' configs.
        if hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder:
            return "encoder-decoder"
        
        # Check for specific architecture types based on common Transformers model classes
        # This is a more robust way to differentiate.
        elif "encoder" in config.architectures[0].lower() and "decoder" in config.architectures[0].lower():
            return "encoder-decoder"
        elif "encoder" in config.architectures[0].lower():
            return "encoder-only"
        elif "decoder" in config.architectures[0].lower() or "lm" in config.architectures[0].lower(): # LM often implies decoder-only
            return "decoder-only"
        
        # Further heuristics based on config attributes
        # Encoder models often have 'num_hidden_layers' or 'num_attention_heads' directly.
        # Decoder models often have 'n_layer' or 'n_head' and a causal attention mechanism.
        elif hasattr(config, "num_hidden_layers") and not hasattr(config, "is_encoder_decoder"):
            # This is a common pattern for encoder-only models like BERT
            if not (hasattr(config, "add_cross_attention") and config.add_cross_attention): # Check for cross-attention, typical in decoders
                return "encoder-only"
        
        elif hasattr(config, "n_layer") and not hasattr(config, "is_encoder_decoder"):
            # This is a common pattern for decoder-only models like GPT-2
            # Decoders also typically have an attention mask that enforces causality
            return "decoder-only"
        
        # Fallback to checking the pipeline tag from the model card metadata
        hf_api = HfApi()
        model_info = hf_api.model_info(model_name)
        if model_info.pipeline_tag:
            if "text2text-generation" in model_info.pipeline_tag or "summarization" in model_info.pipeline_tag or "translation" in model_info.pipeline_tag:
                return "encoder-decoder"
            elif "text-generation" in model_info.pipeline_tag or "causal-lm" in model_info.pipeline_tag:
                return "decoder-only"
            elif "text-classification" in model_info.pipeline_tag or "fill-mask" in model_info.pipeline_tag or "token-classification" in model_info.pipeline_tag:
                return "encoder-only"

        return "unknown"

    except Exception as e:
        print(f"Error processing model {model_name}: {e}")
        return "error"


def get_word_surprisal_csv(scorer:any, BOS, input_csv:str, sentence_col:str, word_col:str, output_csv:str):
    """
    Calculates the surprisal value for a specific word within each sentence in a CSV file and writes the results to a new CSV file.
    Args:
        scorer (any): An object or function used to compute word surprisal.
        BOS: The beginning-of-sentence token or indicator required by the scorer.
        input_csv (str): Path to the input CSV file containing sentences and words.
        sentence_col (str): Name of the column in the input CSV containing sentences.
        word_col (str): Name of the column in the input CSV containing target words.
        output_csv (str): Path to the output CSV file where results will be saved.
    The function reads the input CSV, computes the surprisal for each word in its corresponding sentence,
    adds a new column 'surprisal' to the DataFrame, and writes the updated DataFrame to the output CSV file.
    """
    df = pandas.read_csv(input_csv)
    
    for index, row in df.iterrows():
        sentence = row[sentence_col]
        word = row[word_col]

        # Get the surprisal for the word in the sentence
        surprisal = get_word_surprisal(scorer, BOS, sentence, word)

        # Store the result in the DataFrame
        df.at[index, 'surprisal'] = surprisal
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def get_word_surprisal(scorer:any, BOS:bool, sentence:str, target:str):
    """
    Calculates the surprisal value of a target word within a given sentence using a scorer.

    Args:
        scorer (Any): An object with a `word_score_tokenized` method for scoring words.
        BOS (bool): Whether to include a beginning-of-sentence token.
        sentence (str): The sentence in which to compute word surprisals.
        target (str): The specific word for which to retrieve the surprisal value.

    Returns:
        float or None: The surprisal value of the target word if found, otherwise None.
    """
    surprisals = scorer.word_score_tokenized(
        sentence,
        bos_token=BOS,
        tokenize_function=word_tokenizer,
        surprisal=True,
        bow_correction=True,
    )
    return next((val for word, val in surprisals[0] if word == target), None)


def get_sentence_surprisal_csv(scorer:any, BOS, input_csv:str, sentence_col:str, output_csv:str, sep=","):
    """
    Calculates the surprisal values for sentences in a CSV file and saves the results to a new CSV file.

    Args:
        scorer (any): The scorer object or function used to compute sentence surprisal.
        BOS: The beginning-of-sentence token or value required by the scorer.
        input_csv (str): Path to the input CSV file containing sentences.
        sentence_col (str): Name of the column in the CSV file that contains the sentences.
        filename (str): Path to the output CSV file where results will be saved.
        sep (str, optional): Separator used in the input CSV file. Defaults to ",".

    Returns:
        None: The function saves the updated DataFrame with surprisal values to the specified output file.
    """
    df = pandas.read_csv(input_csv, sep=";")

    for index, row in df.iterrows():
        sentence = row[sentence_col]

        # Get the surprisal for the sentence
        surprisal = get_sentence_surprisal(scorer, BOS, sentence)

        # Store the result in the DataFrame
        df.at[index, 'surprisal'] = surprisal
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)


def get_sentence_surprisal(scorer:any, BOS:bool, sentence:str):
    """
    Calculates the surprisal score of a given sentence using a provided scorer.

    Args:
        scorer (any): An object with a `sequence_score` method that computes the score of a sentence.
        BOS (bool): Whether to include the beginning-of-sentence token in scoring.
        sentence (str): The sentence for which to compute surprisal.

    Returns:
        float or None: The surprisal score (negative log-probability) of the sentence, or None if scoring fails.
    """
    seq_score = scorer.sequence_score(sentence, bos_token=BOS, bow_correction=True)
    return (-1) * seq_score[0] if seq_score is not None else None
