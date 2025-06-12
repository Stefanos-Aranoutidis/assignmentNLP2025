Γίνεται ανακατασκευή δύο προτάσεων, επιλεγμένων από τα 2 κείμενα της εργασίας assignmentNPL2025. Κάνουμε χρήση 3 μοντέλων ανακατασκευής κειμένων:
 Vamsi/T5_Paraphrase_Paws, ramsrigouthamg/t5_paraphraser, eugenesiow/bart-paraphrase

Ο κώδικας  Text_Reconstruction.py:
- Δέχεται ως είσοδο δύο αγγλικά κείμενα (Text 1 και Text 2).
- Τεμαχίζει τα κείμενα σε προτάσεις μέσω `spaCy`.
- Εφαρμόζει παραφράσεις σε κάθε πρόταση με τα τρία παραπάνω μοντέλα.
- Εκτυπώνει τα αποτελέσματα στην κονσόλα, ώστε να γίνει εύκολα η σύγκριση μεταξύ τους.

πρέπει να είναι εγκατεστημένα τα παρακάτω packages:
- pip install transformers
- pip install torch
- pip install spacy
- python -m spacy download en_core_web_sm


 example_of_output.txt: Παράδειγμα εκτέλεσης του κώδικα
