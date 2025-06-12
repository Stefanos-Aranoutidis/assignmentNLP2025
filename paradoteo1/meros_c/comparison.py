from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Text 1
original_text1 = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "Hope you too, to enjoy it as my deepest wishes.",
    "Thank your message to show our words to the doctor, as his next contract checking, to all of us.",
    "I got this message to see the approved message.",
    "In fact, I have received the message from the professor, to show me, this, a couple of days ago.",
    "I am very appreciated the full support of the professor, for our Springer proceedings publication."
]

vamsi_text1 = [
    "Today is our dragon boat festival in our Chinese culture to celebrate it with all safe and great in our lives.",
    "Hope you too, to enjoy it as my deepest wishes.",
    "Thank you for your message to show our words to the doctor, as his next contract checking, to all of us.",
    "I got this message to see the approved message .",
    "In fact, a couple of days ago I received the message from the professor to show me this.",
    "I am very grateful for the full support of the professor for our Springer proceedings publication ."
]

ramsri_text1 = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives.",
    "I hope you too, to enjoy it as my deepest wishes.",
    "Thank you for your message to show our words to the doctor, as his next contract checking, to all of us.",
    "I got this message to see the approved message.",
    "In fact, I have received the message from the professor, to show me, this, a couple of days ago.",
    "I am very appreciated the full support of the professor, for our Springer proceedings publication."
]

bart_text1 = [
    "Today is our dragon boat festival in Chinese culture, to celebrate it with all safe and great in our lives.",
    "Hope you too enjoy it as my deepest wishes.",
    "Thank your message to show our words to the doctor as his next contract checking, to all of us.",
    "I got this message to see the approved message.",
    "In fact, I have received the message from the professor to show me, this, a couple of days ago.",
    "I am very grateful for the full support of the professor for our Springer proceedings publication."
]

# Text 2
original_text2 = [
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?",
    "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.",
    "Because I didn’t see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]

vamsi_text2 = [
    "During our final discussion, I told him about the new submission — the one we were waiting for since last autumn , but the updates were confusing as it did not include the full feedback from reviewer or maybe editor?",
    "Anyway, I believe that the team, although a bit delayed and less communication in recent days, really tried for paper and cooperation best.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link finally came last week, I think .",
    "Please also remind me if the doctor still plan for the acknowledgments section edit before he sends again .",
    "Because I didn’t see that part final yet, or maybe I missed it, I apologize if so .",
    "Let us make sure that all are safe and celebrate the outcome with strong coffee and future targets ."
]

ramsri_text2 = [
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?",
    "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.",
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]

bart_text2 = [
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from the reviewer or maybe editor?",
    "Although bit delay and less communication at recent days, I believe the team really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me if the doctor still plan for the acknowledgments section edit before he sends again.",
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]
def compare_sentences(title, originals, vamsi, ramsri, bart):
    print(f"Semantic Similarity Scores for {title}")
    for i in range(len(originals)):
        original = originals[i]
        score_vamsi = util.cos_sim(model.encode(original), model.encode(vamsi[i]))[0][0].item()
        score_ramsri = util.cos_sim(model.encode(original), model.encode(ramsri[i]))[0][0].item()
        score_bart = util.cos_sim(model.encode(original), model.encode(bart[i]))[0][0].item()

        print(f"Sentence {i+1}")
        print(f"Original: {original}")
        print(f"Vamsi Similarity: {score_vamsi:.3f}")
        print(f"Ramsri Similarity: {score_ramsri:.3f}")
        print(f"BART Similarity : {score_bart:.3f}")
compare_sentences("Text 1", original_text1, vamsi_text1, ramsri_text1, bart_text1)
compare_sentences("Text 2", original_text2, vamsi_text2, ramsri_text2, bart_text2)
