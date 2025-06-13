import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Text 1 + Text 2: Original
texts_original = [
    # Text 1
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "Hope you too, to enjoy it as my deepest wishes.",
    "Thank your message to show our words to the doctor, as his next contract checking, to all of us.",
    "I got this message to see the approved message.",
    "In fact, I have received the message from the professor, to show me, this, a couple of days ago.",
    "I am very appreciated the full support of the professor, for our Springer proceedings publication.",
    # Text 2
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?",
    "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.",
    "Because I didn’t see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets.",
]

# Vamsi
texts_vamsi = [
    "Today is our dragon boat festival in our Chinese culture to celebrate it with all safe and great in our lives.",
    "Hope you too, to enjoy it as my deepest wishes.",
    "Thank you for your message to show our words to the doctor, as his next contract checking, to all of us.",
    "I got this message to see the approved message .",
    "In fact, a couple of days ago I received the message from the professor to show me this.",
    "I am very grateful for the full support of the professor for our Springer proceedings publication .",
    "During our final discussion, I told him about the new submission — the one we were waiting for since last autumn , but the updates were confusing as it did not include the full feedback from reviewer or maybe editor?",
    "Anyway, I believe that the team, although a bit delayed and less communication in recent days, really tried for paper and cooperation best.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link finally came last week, I think .",
    "Please also remind me if the doctor still plan for the acknowledgments section edit before he sends again .",
    "Because I didn’t see that part final yet, or maybe I missed it, I apologize if so .",
    "Let us make sure that all are safe and celebrate the outcome with strong coffee and future targets ."
]

# Ramsri
texts_ramsri = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "I hope you too, to enjoy it as my deepest wishes.",
    "Thank you for your message to show our words to the doctor, as his next contract checking, to all of us.",
    "I got this message to see the approved message.",
    "In fact, I have received the message from the professor, to show me, this, a couple of days ago.",
    "I am very appreciated the full support of the professor, for our Springer proceedings publication.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?",
    "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.",
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]

# BART
texts_bart = [
    "Today is our dragon boat festival in Chinese culture, to celebrate it with all safe and great in our lives.",
    "Hope you too enjoy it as my deepest wishes.",
    "Thank your message to show our words to the doctor as his next contract checking, to all of us.",
    "I got this message to see the approved message.",
    "In fact, I have received the message from the professor to show me, this, a couple of days ago.",
    "I am very grateful for the full support of the professor for our Springer proceedings publication.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from the reviewer or maybe editor?",
    "Although bit delay and less communication at recent days, I believe the team really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me if the doctor still plan for the acknowledgments section edit before he sends again.",
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]

# Συνένωση όλων
all_texts = texts_original + texts_vamsi + texts_ramsri + texts_bart
labels = (["Original"] * len(texts_original) +
          ["Vamsi"] * len(texts_vamsi) +
          ["Ramsri"] * len(texts_ramsri) +
          ["BART"] * len(texts_bart))

# Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_texts)

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Χρώματα
color_map = {
    "Original": "blue",
    "Vamsi": "green",
    "Ramsri": "orange",
    "BART": "red"
}

# Οπτικοποίηση
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.scatter(reduced[i, 0], reduced[i, 1],
                color=color_map[label],
                label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("PCA Visualization of Sentence Embeddings (Original vs Paraphrases)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
