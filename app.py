from flask import Flask, render_template, request, jsonify
import re
import math
import random

# Optional: better paraphrasing with nltk WordNet if installed & data downloaded
USE_NLTK = False
try:
    import nltk
    from nltk.corpus import wordnet as wn
    # If WordNet data not present, this may raise LookupError when used; we'll handle.
    USE_NLTK = True
except Exception:
    USE_NLTK = False

app = Flask(__name__)

# Basic stopwords set
STOPWORDS = {
    "the","and","is","in","it","of","to","a","an","that","this","on","for","with","as","are","was","were",
    "by","be","or","from","at","which","but","not","have","has","had","they","you","we","he","she","his","her"
}

# A small fallback thesaurus for paraphraser (used if nltk wordnet unavailable)
FALLBACK_SYNS = {
    "good": ["great","nice","excellent","solid"],
    "bad": ["poor","subpar","weak"],
    "important": ["crucial","vital","key"],
    "use": ["utilize","employ","apply"],
    "help": ["assist","aid","support"],
    "show": ["display","present","reveal"],
    "change": ["modify","alter","transform"],
    "make": ["create","produce","build"],
    "need": ["require","necessitate"],
    "find": ["discover","locate"],
    "get": ["obtain","acquire"],
    "big": ["large","substantial","considerable"],
    "small": ["tiny","compact","minor"],
    "start": ["begin","commence","initiate"]
}

# Utility: split text to sentences (a simple approach)
def split_sentences(text):
    # keep abbreviations intact as simple heuristic
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# Summarizer: extractive using word frequency scoring
def summarize_text(text, ratio=0.3, min_sentences=1):
    sentences = split_sentences(text)
    if len(sentences) <= min_sentences:
        return sentences

    # build word frequencies
    words = re.findall(r'\w+', text.lower())
    freqs = {}
    for w in words:
        if w in STOPWORDS: continue
        freqs[w] = freqs.get(w, 0) + 1

    # normalize frequencies
    if freqs:
        maxf = max(freqs.values())
        for w in freqs:
            freqs[w] = freqs[w] / maxf

    # score sentences
    sent_scores = []
    for s in sentences:
        s_words = re.findall(r'\w+', s.lower())
        score = 0.0
        for w in s_words:
            score += freqs.get(w, 0)
        # penalize very short sentences slightly
        score = score / (len(s_words) ** 0.5) if s_words else 0
        sent_scores.append((s, score))

    # decide how many sentences to keep
    k = max(min_sentences, int(math.ceil(len(sentences) * ratio)))
    k = min(k, len(sentences))

    # pick top-k sentences preserving original order
    top_sentences = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:k]
    chosen = set(s for s, sc in top_sentences)
    summary = [s for s in sentences if s in chosen]
    return summary

# Paraphrase: sentence-level synonym replacement + slight reordering
def paraphrase_text(text, strength=0.3):
    """
    strength: 0.0-1.0 how aggressive replacement is
    """
    sentences = split_sentences(text)
    paraphrased = []
    for s in sentences:
        tokens = re.findall(r"\w+|\W+", s)  # keep punctuation tokens
        new_tokens = []
        for t in tokens:
            if re.match(r'^\w+$', t):
                # decide whether to replace
                if random.random() < strength:
                    candidate = synonym_for(t.lower())
                    # preserve capitalization
                    if t[0].isupper():
                        candidate = candidate.capitalize()
                    new_tokens.append(candidate)
                else:
                    new_tokens.append(t)
            else:
                new_tokens.append(t)
        new_s = "".join(new_tokens)
        paraphrased.append(new_s)

    # small chance to reorder sentences for variety
    if len(paraphrased) > 1 and random.random() < 0.2 * strength:
        random.shuffle(paraphrased)
    return " ".join(paraphrased)

def synonym_for(word):
    # Try WordNet if available
    if USE_NLTK:
        try:
            synsets = wn.synsets(word)
            # choose lemma from first synset that's not the same word, prefer single tokens
            for s in synsets:
                for lemma in s.lemmas():
                    name = lemma.name().replace('_',' ')
                    if name.lower() != word and ' ' not in name:
                        return name
            # fallback to lemma names of first synset
            if synsets:
                name = synsets[0].lemmas()[0].name().replace('_',' ')
                return name
        except LookupError:
            # missing data - fall through to fallback
            pass
        except Exception:
            pass

    # fallback thesaurus
    if word.lower() in FALLBACK_SYNS:
        return random.choice(FALLBACK_SYNS[word.lower()])
    # tiny morphological attempt: try to handle plural or -ing
    if word.endswith("ing") and len(word) > 4:
        return word[:-3]  # naive
    return word  # default: unchanged

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def api_summarize():
    data = request.json or {}
    text = data.get('text', '').strip()
    ratio = float(data.get('ratio', 0.3))
    min_sentences = int(data.get('min_sentences', 1))
    if not text:
        return jsonify({"ok": False, "error": "No text provided."}), 400
    summary_sentences = summarize_text(text, ratio=ratio, min_sentences=min_sentences)
    summary = " ".join(summary_sentences)
    return jsonify({"ok": True, "summary": summary, "sentences": summary_sentences})

@app.route('/paraphrase', methods=['POST'])
def api_paraphrase():
    data = request.json or {}
    text = data.get('text', '').strip()
    strength = float(data.get('strength', 0.3))
    if not text:
        return jsonify({"ok": False, "error": "No text provided."}), 400
    para = paraphrase_text(text, strength=strength)
    return jsonify({"ok": True, "paraphrase": para})

if __name__ == '__main__':
    print("Starting summarizer/paraphraser on http://127.0.0.1:5000")
    if USE_NLTK:
        print("NLTK WordNet support: available")
    else:
        print("NLTK WordNet support: NOT available (using fallback synonyms)")
    app.run(debug=True)
