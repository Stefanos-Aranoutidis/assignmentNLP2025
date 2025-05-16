from transformers import pipeline

paraphraser_bart = pipeline("text2text-generation", model="facebook/bart-large-cnn")
paraphraser_t5 = pipeline("text2text-generation", model="t5-small")
paraphraser_pegasus = pipeline("text2text-generation", model="google/pegasus-xsum")

#Αρχικά κείμενα
sentence_1 = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
sentence_2 = "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"

#BART
result_bart_1 = paraphraser_bart(sentence_1, max_length=60, do_sample=False)[0]['generated_text']
result_bart_2 = paraphraser_bart(sentence_2, max_length=80, do_sample=False)[0]['generated_text']

#T5
result_t5_1 = paraphraser_t5(sentence_1, max_length=60, do_sample=False)[0]['generated_text']
result_t5_2 = paraphraser_t5(sentence_2, max_length=80, do_sample=False)[0]['generated_text']

#PEGASUS
result_pegasus_1 = paraphraser_pegasus(sentence_1, max_length=60, do_sample=False)[0]['generated_text']
result_pegasus_2 = paraphraser_pegasus(sentence_2, max_length=80, do_sample=False)[0]['generated_text']

# Εκτύπωση αποτελεσμάτων
print("Original Sentence 1:\n", sentence_1)
print("Reconstructed (BART):\n", result_bart_1)
print("Reconstructed (T5):\n", result_t5_1)
print("Reconstructed (PEGASUS):\n", result_pegasus_1, "\n")

print("Original Sentence 2:\n", sentence_2)
print("Reconstructed (BART):\n", result_bart_2)
print("Reconstructed (T5):\n", result_t5_2)
print("Reconstructed (PEGASUS):\n", result_pegasus_2)
