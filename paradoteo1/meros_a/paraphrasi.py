import re

# Custom λεξικό μετατροπών (rule-based)
synonym_dict = {
    "bit delay": "some delays",
    "less communication": "limited communication",
    "tried best": "did their best",
    "Hope you too, to enjoy it as my deepest wishes.":
        "I also hope that you enjoy it — these are my sincerest wishes.",
}


# Συνάρτηση ανακατασκευής πρότασης με απλό rule-based σύστημα
def reconstruct_sentence(sentence: str) -> str:
    sentence = sentence.strip()

    # Αν η πρόταση είναι σε λεξικό μας, επιστρέφουμε έτοιμη ανακατασκευή
    if sentence in synonym_dict:
        return synonym_dict[sentence]

    # Απλές αντικαταστάσεις φράσεων
    for phrase, replacement in synonym_dict.items():
        if phrase in sentence:
            sentence = sentence.replace(phrase, replacement)

    # Βασική ορθογραφική εξομάλυνση
    sentence = re.sub(r'\bi\b', 'I', sentence)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith('.'):
        sentence += '.'

    return sentence


# Παραδείγματα από τα 2 κείμενα
s1 = "Hope you too, to enjoy it as my deepest wishes."
s2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

# Εφαρμογή
print("Original:", s1)
print("Reconstructed:", reconstruct_sentence(s1))
print("\nOriginal:", s2)
print("Reconstructed:", reconstruct_sentence(s2))

