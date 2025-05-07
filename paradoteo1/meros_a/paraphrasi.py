from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Αρχικές προτάσεις
sentence_1 = "I am very appreciated the full support of the professor, for our Springer proceedings publication"
sentence_2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

result_1 = paraphraser(sentence_1, max_length=60, do_sample=False)[0]['generated_text']
result_2 = paraphraser(sentence_2, max_length=80, do_sample=False)[0]['generated_text']

# Εκτύπωση αποτελεσμάτων
print("🔹 Original Sentence 1:\n", sentence_1)
print("✅ Reconstructed:\n", result_1, "\n")

print("🔹 Original Sentence 2:\n", sentence_2)
print("✅ Reconstructed:\n", result_2)
