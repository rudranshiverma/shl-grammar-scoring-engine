# Grammar Scoring Engine for Spoken English

SHL Hiring Assessment 2026 (Kaggle). Predicts a 0–5 grammar score for 45–60 s spoken answers. 769 training clips, 216 test clips. Metrics: RMSE and Pearson correlation.

## Results

| Metric | Value |
| :--- | :--- |
| 5-fold CV RMSE | 0.514 |
| 5-fold CV Pearson | 0.912 |
| Training RMSE (in-sample) | 0.182 |
| Public leaderboard RMSE | 0.389 |

CV metrics are computed on out-of-fold predictions, with folds stratified by score band and clip length.

## Approach

1. **Transcription.** Whisper large-v3, prompted with a disfluent example so it keeps fillers, repetitions and false starts. Word timestamps are kept for fluency features.
2. **Rubric-aligned features.** CoLA grammaticality (RoBERTa), spaCy clause complexity and sentence completeness, disfluency rates, fluency from timestamps, and Whisper's confidence. All are rates, because test clips are shorter than training clips.
3. **Text embedding.** RoBERTa hidden states of the transcript, mean-pooled.
4. **Audio embeddings.** WavLM-large and the Whisper-large-v3 encoder, mean and standard deviation over time per layer. Layers are chosen by cross-validation.
5. **Model.** Equal-weight average of four models: ridge on each audio embedding, and LightGBM on features plus embeddings, with and without the Whisper embedding.

## Progression

| Step | CV RMSE |
| :--- | :--- |
| Predict the training mean | 1.238 |
| LightGBM on handcrafted features | 1.027 |
| Ridge on RoBERTa transcript embedding | 0.909 |
| Ridge on WavLM-large embedding | 0.535 |
| + LightGBM on all inputs, blended | 0.518 |
| + Whisper encoder models (final) | 0.514 |

Audio embeddings cut the error roughly in half compared with anything text-based: the raters' grammar scores reflect how the speech sounds, not only the words.

## Repository

```
notebooks/
  01_transcription.ipynb               Whisper transcripts and word timings
  02_wavlm_embeddings.ipynb            WavLM-large audio embeddings
  03_whisper_encoder_embeddings.ipynb  Whisper encoder audio embeddings
  04_grammar_scoring_model.ipynb       Features, models, evaluation, report, submission
requirements.txt
```

## Running

Built for Kaggle notebooks.

1. Run notebooks 01–03 with the competition data attached (GPU T4 x2, Internet on). Runtimes: about 1 h 45 min, 25 min and 25 min.
2. Attach the competition data and the outputs of 01–03 to notebook 04 and run all cells. It writes `submission.csv`.

Competition data, embeddings and the submission file are not included.

## Limitations

- Repeated speakers in the training set were not checked. If they exist, random folds overestimate performance on new speakers.
- Short (~45 s) clips make up 69% of the test set and are predicted less accurately (CV RMSE 0.59 vs 0.49 for long clips).
- Tried and dropped: LanguageTool error rates, PCA-compressed embeddings, the CrisperWhisper encoder, and LLM rubric grading (Qwen2.5-7B). Results are in notebook 04, Section 10.
