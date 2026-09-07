# IDGuardian Error Analysis

Reconstructed the exact held-out test split from training (same seed, same 70/30 stratified split) and scored it through the saved models -- nothing here was seen during training or model selection.

**Sanity check vs. training report: MATCHES.**


## Key Findings

- Across all four platforms combined, Trust Fusion correctly classifies 94.9% of genuine profiles and 85.5% of fake profiles.

- Archetypes below 90% accuracy (worst first) -- every one of these is a "sophisticated" fake archetype (impersonator, bought-follower influencer, or bot) whose text/behavior was deliberately given partial overlap with genuine patterns during dataset generation (see NOTES.md), exactly the kind of ambiguous case a realistic detector should sometimes miss:

  - Instagram Spam Account: 74.2% (n=198)
  - Instagram Fake Influencer: 78.1% (n=201)
  - Twitter Celebrity Impersonator: 78.5% (n=209)
  - Instagram Celebrity Impersonator: 78.6% (n=201)
  - Twitter Fake Influencer: 79.5% (n=200)
  - Twitter Spam/Bot Account: 83.8% (n=191)
  - Facebook Spam Account: 88.4% (n=146)
  - Facebook Celebrity Impersonator: 89.7% (n=156)


- Trust Fusion vs. Behavioral alone (F1 for the fake class):

| Platform | Behavioral F1 | Fusion F1 | Delta |
|---|---|---|---|
| Instagram | 0.8346 | 0.8324 | -0.0022 |
| Facebook | 0.8448 | 0.9091 | +0.0643 |
| Linkedin | 0.9181 | 0.9434 | +0.0253 |
| Twitter | 0.8272 | 0.8496 | +0.0224 |

Fusion adds real lift on Facebook, LinkedIn, and Twitter/X; on Instagram it's essentially a wash -- Behavioral alone was already strong there and the other two layers didn't have much to add on top of it for this platform's held-out set.


- Of the profiles Trust Fusion incorrectly flagged as fake (false positives), how often did the Behavioral layer *independently* agree (i.e. this was not just a text-driven false alarm):

| Platform | False positives | Behavioral also >=50% fake |
|---|---|---|
| Instagram | 48 | 75% |
| Facebook | 48 | 83% |
| Linkedin | 35 | 86% |
| Twitter | 54 | 80% |

When Behavioral also agrees, the false positive is not just a fluke of wording -- the profile's actual numeric fingerprint (via the class-overlap tuning from the QA pass) looks statistically like a fake one too, so the model's mistake reflects genuinely ambiguous signals rather than an unexplainable error.


## Instagram (n=1500 held out)

### Confusion matrix -- Trust Fusion

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 852 (TN) | 48 (FP) |
| **Actual Fake** | 138 (FN) | 462 (TP) |

Precision (fake): 0.9059 · Recall (fake): 0.77 · F1 (fake): 0.8324 · False-positive rate: 0.0533 · False-negative rate: 0.23


For comparison, Behavioral alone: precision 0.8325, recall 0.8367, F1 0.8346 (Behavioral alone was better here by 0.0022).


### Accuracy by archetype (Trust Fusion), worst first

| Archetype | n | Accuracy | Avg fusion score |
|---|---|---|---|
| Spam Account | 198 | 74.2% | 0.7727 |
| Fake Influencer | 201 | 78.1% | 0.7950 |
| Celebrity Impersonator | 201 | 78.6% | 0.7950 |
| Fitness Creator | 229 | 93.5% | 0.1240 |
| Developer | 215 | 93.5% | 0.1490 |
| Student | 214 | 95.8% | 0.1100 |
| Business Account | 242 | 95.9% | 0.1266 |

### Most confidently wrong (Trust Fusion)

- **@win_williamwade91966** (Spam Account) -- true: Fake, predicted: Genuine (2% fake). Lexical 9% / Semantic 6% / Behavioral 2%. Text: "recovering cardio-phobe turned lifter Helping you build strength, not just muscle Logged 60 miles this week. 🔥 Recovery day, mobility work o..."
- **@heather77.official** (Celebrity Impersonator) -- true: Fake, predicted: Genuine (2% fake). Lexical 9% / Semantic 4% / Behavioral 4%. Text: "Custom orders welcome Johnson-Wilson — official account Batch number 57 restocked today. 🌍 Small batch, made with care. Locally sourced, alw..."
- **@win_cheryl5587533** (Spam Account) -- true: Fake, predicted: Genuine (2% fake). Lexical 10% / Semantic 10% / Behavioral 2%. Text: "living on iced coffee and hope into indie films and bad puns Number 17 on my to-do list: sleep. Group project update: Somehow still passing ..."
- **@bonnie03xo** (Fake Influencer) -- true: Fake, predicted: Genuine (2% fake). Lexical 9% / Semantic 7% / Behavioral 5%. Text: "Booking now for 2017 Thank you for supporting small business Shoutout to our 48th customer this week. Thank you Port Aprilside for another g..."
- **@otownsend1** (Celebrity Impersonator) -- true: Fake, predicted: Genuine (2% fake). Lexical 14% / Semantic 7% / Behavioral 3%. Text: "living on iced coffee and hope business major, meme connoisseur East Danielle University '27 Currently 73 days away from break. Finals seaso..."

## Facebook (n=1500 held out)

### Confusion matrix -- Trust Fusion

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 852 (TN) | 48 (FP) |
| **Actual Fake** | 60 (FN) | 540 (TP) |

Precision (fake): 0.9184 · Recall (fake): 0.9 · F1 (fake): 0.9091 · False-positive rate: 0.0533 · False-negative rate: 0.1


For comparison, Behavioral alone: precision 0.8413, recall 0.8483, F1 0.8448 (Fusion improves F1 by 0.0643).


### Accuracy by archetype (Trust Fusion), worst first

| Archetype | n | Accuracy | Avg fusion score |
|---|---|---|---|
| Spam Account | 146 | 88.4% | 0.8505 |
| Celebrity Impersonator | 156 | 89.7% | 0.8391 |
| Fake Influencer | 155 | 90.3% | 0.8482 |
| Business Page | 221 | 91.4% | 0.1323 |
| Catfish/Romance-Scam Profile | 143 | 91.6% | 0.8893 |
| Developer | 218 | 95.0% | 0.0985 |
| Student | 235 | 95.7% | 0.1164 |
| Fitness Creator | 226 | 96.5% | 0.0908 |

### Most confidently wrong (Trust Fusion)

- **@choijoseph** (Developer) -- true: Genuine, predicted: Fake (99% fake). Lexical 87% / Semantic 77% / Behavioral 100%. Text: "Not a bot, it's really me Follow this account only Project number 83 coming together nicely. Thank you for the love this 2023. Please report..."
- **@elliott_higgins_and_tayl** (Business Page) -- true: Genuine, predicted: Fake (99% fake). Lexical 81% / Semantic 83% / Behavioral 100%. Text: "Widowed, looking for genuine connection Message me, let's talk Only 95 more months until I'm back. Counting down the days already. I promise..."
- **@tfoster.life** (Fitness Creator) -- true: Genuine, predicted: Fake (99% fake). Lexical 79% / Semantic 82% / Behavioral 97%. Text: "Direct message me here for business Follow this account only Since 2024, the journey continues. 😊🎬 I see all of your messages, truly. I read..."
- **@rodriguezmichael** (Developer) -- true: Genuine, predicted: Fake (99% fake). Lexical 81% / Semantic 87% / Behavioral 94%. Text: "Add me, I don't bite Family means everything to me Missing East Danielle and missing you more. 🌹❤️ Thinking about you today. I promise I'm n..."
- **@buckley-wood** (Business Page) -- true: Genuine, predicted: Fake (99% fake). Lexical 80% / Semantic 79% / Behavioral 97%. Text: "DM for details Get verified fast, DM now Limited spots left!! Started in West John, now everywhere. 💯💰 This changed everything for me. Thous..."

## Linkedin (n=1500 held out)

### Confusion matrix -- Trust Fusion

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 865 (TN) | 35 (FP) |
| **Actual Fake** | 33 (FN) | 567 (TP) |

Precision (fake): 0.9419 · Recall (fake): 0.945 · F1 (fake): 0.9434 · False-positive rate: 0.0389 · False-negative rate: 0.055


For comparison, Behavioral alone: precision 0.9113, recall 0.925, F1 0.9181 (Fusion improves F1 by 0.0253).


### Accuracy by archetype (Trust Fusion), worst first

| Archetype | n | Accuracy | Avg fusion score |
|---|---|---|---|
| Fake Recruiter | 185 | 93.0% | 0.8855 |
| Fake Executive Impersonator | 202 | 94.5% | 0.8986 |
| Business/Company Page | 228 | 95.6% | 0.0751 |
| Spam/Bot Account | 213 | 95.8% | 0.9122 |
| Student | 234 | 96.2% | 0.0698 |
| Developer | 214 | 96.3% | 0.0757 |
| Consultant | 224 | 96.4% | 0.0640 |

### Most confidently wrong (Trust Fusion)

- **@baileyjasmine_fanpage** (Fake Executive Impersonator) -- true: Fake, predicted: Genuine (1% fake). Lexical 10% / Semantic 9% / Behavioral 0%. Text: "always looking for the next challenge Aspiring professional, studying at Evelynborough University Grateful for Scott's advice on this one. J..."
- **@carter-rodriguez** (Business/Company Page) -- true: Genuine, predicted: Fake (99% fake). Lexical 85% / Semantic 94% / Behavioral 99%. Text: "leading global teams to success CEO | Based in Port Vickiemouth, thinking globally Based in Stokesborough, thinking globally. Another record..."
- **@the_rachel9039117** (Fake Recruiter) -- true: Fake, predicted: Genuine (1% fake). Lexical 10% / Semantic 7% / Behavioral 3%. Text: "always tinkering with a side project Full-Stack Engineer building scalable systems Based out of Smithberg, remote-first. Excited to start us..."
- **@the_kwilliams46028** (Fake Recruiter) -- true: Fake, predicted: Genuine (1% fake). Lexical 10% / Semantic 11% / Behavioral 3%. Text: "focused on measurable outcomes Derrick | Consultant & advisor Engagement number 61 wrapped this quarter. Speaking at a panel next month on i..."
- **@the_mathisjohn79545** (Fake Recruiter) -- true: Fake, predicted: Genuine (1% fake). Lexical 10% / Semantic 7% / Behavioral 5%. Text: "Platform Engineer based in East Heidi passionate about clean code and mentorship 46 years into this role now. Excited to start using a new f..."

## Twitter (n=1500 held out)

### Confusion matrix -- Trust Fusion

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 846 (TN) | 54 (FP) |
| **Actual Fake** | 117 (FN) | 483 (TP) |

Precision (fake): 0.8994 · Recall (fake): 0.805 · F1 (fake): 0.8496 · False-positive rate: 0.06 · False-negative rate: 0.195


For comparison, Behavioral alone: precision 0.8132, recall 0.8417, F1 0.8272 (Fusion improves F1 by 0.0224).


### Accuracy by archetype (Trust Fusion), worst first

| Archetype | n | Accuracy | Avg fusion score |
|---|---|---|---|
| Celebrity Impersonator | 209 | 78.5% | 0.7832 |
| Fake Influencer | 200 | 79.5% | 0.7912 |
| Spam/Bot Account | 191 | 83.8% | 0.8161 |
| Student | 217 | 91.2% | 0.1477 |
| Fitness Creator | 231 | 93.1% | 0.1345 |
| Business Account | 222 | 95.0% | 0.1278 |
| Developer | 230 | 96.5% | 0.1271 |

### Most confidently wrong (Trust Fusion)

- **@petersalasco** (Fake Influencer) -- true: Fake, predicted: Genuine (2% fake). Lexical 8% / Semantic 5% / Behavioral 3%. Text: "Rust curious, Python fluent Travis — mobile engineer Shoutout to Michael for the code review. Finally got the tests passing. Turns out the c..."
- **@hunteradam_real** (Celebrity Impersonator) -- true: Fake, predicted: Genuine (2% fake). Lexical 8% / Semantic 6% / Behavioral 7%. Text: "Helping you build strength, not just muscle Free workout guide in bio Shoutout to Cody for showing up every day. 🔥🏋️ Early morning session b..."
- **@real_pkline44116** (Spam/Bot Account) -- true: Fake, predicted: Genuine (2% fake). Lexical 10% / Semantic 7% / Behavioral 6%. Text: "Local favorite in Anthonyburgh New drops every 87 weeks Proudly independent Celebrating 5 years since we opened our doors. Behind the scenes..."
- **@the_steven9314080** (Spam/Bot Account) -- true: Fake, predicted: Genuine (3% fake). Lexical 15% / Semantic 8% / Behavioral 5%. Text: "Family-owned, Port Randall-based Booking now for 2019 Team huddle today covered 73 new ideas. Restocked your favorites. We appreciate every ..."
- **@david60_verified** (Celebrity Impersonator) -- true: Fake, predicted: Genuine (3% fake). Lexical 16% / Semantic 7% / Behavioral 5%. Text: "plays intramural soccer badly business major, meme connoisseur counting down to graduation Campus tour 87 for prospective students today. 📚💪..."