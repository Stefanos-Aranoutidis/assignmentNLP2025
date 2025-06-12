from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

# Κείμενα Εισόδου
text1 = """
Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.
Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message.
In fact, I have received the message from the professor, to show me, this, a couple of days ago.
I am very appreciated the full support of the professor, for our Springer proceedings publication.
"""

text2 = """
During our final discuss, I told him about the new submission — the one we were waiting since last autumn,
but the updates was confusing as it not included the full feedback from reviewer or maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.
We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.
Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets.
"""


nlp = spacy.load("en_core_web_sm")
def split_sentences(text):
    return [sent.text.strip() for sent in nlp(text).sents]

# Μοντέλο 1: Vamsi
tok1 = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
mod1 = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

def paraphrase_1(sentence):
    prompt = f"paraphrase: {sentence}"
    inputs = tok1(prompt, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
    output = mod1.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], num_beams=5, max_length=256)
    return tok1.decode(output[0], skip_special_tokens=True)

# Μοντέλο 2: ramsrigouthamg
tok2 = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
mod2 = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")

def paraphrase_2(sentence):
    prompt = f"paraphrase: {sentence} </s>"
    inputs = tok2(prompt, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
    output = mod2.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], num_beams=5, max_length=256)
    return tok2.decode(output[0], skip_special_tokens=True)

# Μοντέλο 3: BART
tok3 = AutoTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
mod3 = AutoModelForSeq2SeqLM.from_pretrained("eugenesiow/bart-paraphrase")

def paraphrase_3(sentence):
    inputs = tok3(sentence, return_tensors="pt", max_length=256, truncation=True)
    output = mod3.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], num_beams=5, max_length=256)
    return tok3.decode(output[0], skip_special_tokens=True)

# Εκτέλεση
def run_pipeline(text, label):
    print(f"Ανακατασκευή {label} ")
    for sent in split_sentences(text):
        print(f"Original: {sent}")
        print(f"Vamsi:      {paraphrase_1(sent)}")
        print(f"Ramsri:     {paraphrase_2(sent)}")
        print(f"BART:       {paraphrase_3(sent)}")

# Εκτέλεση
run_pipeline(text1, "Text 1")
run_pipeline(text2, "Text 2")
