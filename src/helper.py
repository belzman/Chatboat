import re

def normalize_amharic(text):
    """
    Normalizes Amharic homophones AND vowel inconsistencies.
    Includes the 'የ' vs 'ዬ' variations to ensure search consistency.
    """
    if not text:
        return ""
        
    rep_map = {
        # 1. Standard Homophones (The 'h', 's', 'a', 'ts' families)
        'ሐ': 'ሀ', 'ኀ': 'ሀ', 'ሃ': 'ሀ', 'ሓ': 'ሀ', 'ኃ': 'ሀ', 'ኅ': 'ሀ','ሑ':'ሁ','ኁ':'ሁ','ሒ':'ሂ','ኂ':'ሂ',
        'ሔ':'ሄ','ኄ':'ሄ','ሕ':'ህ','ኅ':'ህ','ሖ':'ሆ','ኆ':'ሆ',
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
        'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'አ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
        'ጸ': 'ፀ', 'ጹ': 'ፁ', 'ጺ': 'ጲ', 'ጻ': 'ፃ', 'ጼ': 'ፄ', 'ፅ': 'ፅ', 'ጾ': 'ፆ',
        
        # 2. Vowel-level Variations (Special case: 'የ' variations)
        'ዬ': 'የ', 'ዪ': 'ይ', 
        
        # 3. Labialized variations (Common in suffixes)
        'ቷ': 'ቱዋ', 'ኗ': 'ኑዋ', 'ሏ': 'ሉዋ', 'ሟ': 'ሙዋ', 'ሯ': 'ሩዋ', 'ቷል': 'ቱዋል'
    }
    
    for char, rep in rep_map.items():
        text = text.replace(char, rep)
    return text

def preprocess_text(text):
    """
    Optimized for Bilingual Medical Chatbot (Text + Voice).
    Prepares raw text input for the Vector Database (ingest.py & retrieval in que.py).
    """
    if not text:
        return ""

    # Clean URLs/Emails first
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Apply the deep Amharic normalization
    text = normalize_amharic(text)

    # Convert Ethiopic full stop '።' to English '.' for the TextSplitter
    text = text.replace('።', '.').replace('፡', ' ')
    
    # Handle English Lowercasing safely
    text = "".join([char.lower() if 'A' <= char <= 'Z' else char for char in text])

    # Keep numbers (crucial for medical dosages) but clean symbols
    text = re.sub(r'[*"/$%#?@^&~_{}\[\]]', '', text)

    # Normalize whitespace structures
    text = " ".join(text.split())

    return text.strip()

def stem_amharic_word(token):
    """
    Lightweight Rule-Based Amharic Stemmer (Affix Stripper).
    Iteratively strips common prefixes and suffixes from inflected Amharic words 
    to extract the underlying semantic root for accurate BLEU calculation.
    """
    # If the token is too short or contains non-Amharic characters, do not stem
    if len(token) <= 2 or not re.match(r'^[\u1200-\u137F]+$', token):
        return token

    # 1. Common Suffixes (Ordered by length to strip longest matches first)
    # Includes plural markers, possessives, and definitives (-ዎቻቸውን, -ቸውን, -ው, -ዋ, etc.)
    suffixes = [
        'ዎቻቸውን', 'ዎቻችሁ', 'ዎቻቸውን', 'ዎቻችን', 'ያቸውን', 'ያችሁ', 'ያችን', 
        'ቸውን', 'ችሁን', 'ችንን', 'ዎቹን', 'ዋን', 'ውን', 'ውማ', 'ውኑ',
        'ቸው', 'ችሁ', 'ችን', 'ዎቹ', 'ዋት', 'ዎት', 'ዊት', 'ነት', 'ኛ', 'ዊ',
        'ው', 'ዋ', 'ኡ', 'ኢ', 'ን', 'ም', 'ስ', 'ህ', 'ሽ', 'ሌ'
    ]
    
    # 2. Common Prefixes
    # Includes prepositions and conjunctions (የማይ-, ስለ-, እንደ-, ለማ-, ለ-, በ-, ከ-, የ-, ወደ-)
    prefixes = [
        'የማይ', 'ስለ', 'እንደ', 'ለማ', 'በማ', 'ከማ', 'የ', 'ለ', 'በ', 'ከ', 'ወደ', 'ማ','አስ'
    ]

    # Iterative Suffix Stripping
    for suf in suffixes:
        if token.endswith(suf) and len(token) - len(suf) >= 2:
            token = token[:-len(suf)]
            break  # Strip one major suffix block

    # Iterative Prefix Stripping
    for pref in prefixes:
        if token.startswith(pref) and len(token) - len(pref) >= 2:
            token = token[len(pref):]
            break  # Strip one major prefix block

    return token

def tokenize_to_words(text):
    """
    Advanced Stemming-Enabled Tokenizer used by evaluate.py and que.py.
    Normalizes text, strips bilingual punctuation, tokenizes, and stems Amharic 
    words so morphological variants match perfectly in lexical evaluation.
    """
    if not isinstance(text, str):
        text = str(text)
        
    # Clean URLs, Emails, and apply standard homophone normalization
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = normalize_amharic(text)
    
    # Lowercase text for fair English matching
    normalized = text.lower()
    
    # Clean markdown and formatting symbols
    normalized = normalized.replace("\n", " ").replace("*", " ").replace("#", " ")
    normalized = re.sub(r'[።፣፤፡\.!\?,;:_\-\(\)\[\]\{\}\"\']', ' ', normalized)
    
    # Initial word splitting by space
    raw_tokens = [word.strip() for word in normalized.split() if word.strip()]
    
    processed_tokens = []
    for token in raw_tokens:
        # Check if token contains Amharic characters
        if re.search(r'[\u1200-\u137F]', token):
            stemmed = stem_amharic_word(token)
            processed_tokens.append(stemmed)
        else:
            # Keep English words as they are (lowercased)
            processed_tokens.append(token)
            
    return processed_tokens