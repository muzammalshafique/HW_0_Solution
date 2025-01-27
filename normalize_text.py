import argparse
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text, lower, stem, lemma, remove_stopwords, custom_option, remove_urls, min_length):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Lowercasing
    if lower:
        tokens = [token.lower() for token in tokens]
    
    # Remove URLs
    if remove_urls:
        tokens = [re.sub(r'http[s]?://\S+|www\.\S+', '', token) for token in tokens]
        tokens = [token for token in tokens if token]  # Remove empty tokens
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    if lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Custom option: Remove numbers
    if custom_option:
        tokens = [token for token in tokens if not token.isdigit()]
    
    # Filter by token length
    tokens = [token for token in tokens if len(token) >= min_length]
    
    return tokens

def plot_word_frequencies(token_counts, top_n=10, log_scale=False, graph_type="bar"):
    """
    Plot a bar or line graph for token frequencies.
    
    Args:
        token_counts (dict): A dictionary of tokens and their counts.
        top_n (int): Number of top words to visualize.
        log_scale (bool): Whether to use a logarithmic scale for axes.
        graph_type (str): Type of graph to plot ("bar" or "line").
    """
    # Sort tokens by frequency and select the top N
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, frequencies = zip(*sorted_tokens)  # Split into words and frequencies
    
    # Create x-axis ranks
    ranks = np.arange(1, len(words) + 1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    if graph_type == "bar":
        plt.bar(ranks, frequencies, color='skyblue', alpha=0.8, label="Frequency")
    elif graph_type == "line":
        plt.plot(ranks, frequencies, marker='o', linestyle='-', color='skyblue', label="Frequency")
    else:
        print(f"Error: Invalid graph type '{graph_type}'. Defaulting to bar graph.")
        plt.bar(ranks, frequencies, color='skyblue', alpha=0.8, label="Frequency")
    
    # Set log scale if enabled
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
    
    # Add labels, title, and grid
    plt.xticks(ranks, words, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Rank (Word)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Word Frequency Distribution', fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    # Show plot
    plt.legend()
    plt.show()

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Normalize and count tokens in a text file.")
    parser.add_argument("file", type=str, help="Path to the text file.")
    parser.add_argument("--lower", action="store_true", help="Convert text to lowercase.")
    parser.add_argument("--stem", action="store_true", help="Apply stemming to tokens.")
    parser.add_argument("--lemma", action="store_true", help="Apply lemmatization to tokens.")
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove stopwords from the text.")
    parser.add_argument("--custom-option", action="store_true", help="Apply custom option (e.g., remove numbers).")
    parser.add_argument("--remove-urls", action="store_true", help="Remove URLs from the text.")
    parser.add_argument("--min-length", type=int, default=2, help="Minimum token length.")
    parser.add_argument("--freq-threshold", type=int, default=0, help="Frequency threshold for tokens.")
    parser.add_argument("--visualize", action="store_true", help="Visualize top word frequencies.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top words to visualize.")
    parser.add_argument("--graph", type=str, choices=["bar", "line"], default="bar", help="Type of graph ('bar' or 'line').")

    args = parser.parse_args()
    
    # Read the file
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return
    
    # Preprocess the text
    tokens = preprocess_text(
        text, 
        lower=args.lower, 
        stem=args.stem, 
        lemma=args.lemma, 
        remove_stopwords=args.remove_stopwords, 
        custom_option=args.custom_option,
        remove_urls=args.remove_urls,
        min_length=args.min_length
    )
    
    # Count token frequencies
    token_counts = Counter(tokens)
    
    # Apply frequency threshold
    if args.freq_threshold > 0:
        token_counts = {word: count for word, count in token_counts.items() if count > args.freq_threshold}
    
    # Sort by frequency (descending order)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the top N words
    print(f"Top {args.top_n} words by frequency:")
    for token, count in sorted_tokens[:args.top_n]:
        print(f"{token}: {count}")

    # Print the last 10 words
    print("\nLast 10 words by frequency:")
    for token, count in sorted_tokens[-10:]:
        print(f"{token}: {count}")
    
    # Plot word frequencies if visualize option is enabled
    if args.visualize:
        plot_word_frequencies(token_counts, top_n=args.top_n, log_scale=False, graph_type=args.graph)

if __name__ == "__main__":
    main()