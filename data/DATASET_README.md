# Fatigue Detection Challenge Dataset

280 audio recordings from 14 participants, each speaking under two conditions:
before a work shift (`non_fatigued`) and after a work shift (`fatigued`). Each
participant has recordings from both conditions across multiple sessions.

## Files

- `audio/` — 280 WAV files (16kHz, mono), named `<participant_id>_s<session>_<activity>.wav`
- `labels.csv` — columns: `filename, participant_id, session_number, activity_type, label`

`activity_type` is one of:
- `read_aloud` — reading a short passage aloud
- `read_aloud_nonsense` — reading a scrambled version of the same passage
- `describe_picture` — describing an image in spontaneous speech

## The challenge

Train the best binary classifier (fatigued vs. non_fatigued) you can.

## One important rule, not optional

**Evaluate with grouped, speaker-independent cross-validation — never a random
train/test split.** Every participant contributes many recordings, and a
random split will leak the same person's voice into both train and test,
which inflates accuracy without teaching you anything about whether your
model generalizes to a new person. Use `participant_id` as the group key:
- Leave-one-participant-out (LOSO), or
- Grouped k-fold (e.g. `sklearn.model_selection.GroupKFold`) keyed on `participant_id`

A held-out random split will look great and mean nothing. I will ask specifically how you split the data before trusting your
number.
