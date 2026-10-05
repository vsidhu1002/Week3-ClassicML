# Assignment 3: Speech Emotion Recognition

GitHub account: **vsidhu1002**. Experiment: SVM RBF, KNN, Random Forest and additional Logistic Regression, each raw and standard scaled. `Assignment3_Results.ipynb` contains actual executed outputs; `run_experiment.py` implements the full reproducible experiment. The original tutorial is retained unchanged.

## Results
- RAVDESS: 1,440 clips, stratified 80/20 split (1,152 training / 288 test), random seed 42.
- Personal: eight newly recorded clips, one per emotion, saying "I'm going to see my friend tomorrow." These were recorded for this assignment, not the uncompleted previous assignment.
- Best RAVDESS accuracy: scaled Random Forest, 55.2%.
- Best personal accuracy: raw Random Forest, 37.5% (3/8).
- Metrics, confusion matrices, feature summaries, predictions, versions and exact split manifest are under `results/`.

## Run locally or in Colab
1. Install Python requirements: `pip install -r requirements.txt`.
2. Download the [RAVDESS speech archive](https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1), extract into `data/ravdess/` (actor subfolders may be nested).
3. Place eight mono WAV files named `angry.wav`, `calm.wav`, `disgust.wav`, `fearful.wav`, `happy.wav`, `neutral.wav`, `sad.wav`, `surprised.wav` in `personal/`.
4. Run `python run_experiment.py --ravdess data/ravdess --personal personal --output results --jobs 4`.
5. View `Assignment3_Results.ipynb` or the PDF report. Cached feature arrays are validated against filenames; delete the cache if you intentionally change extraction settings or file bytes.

To convert phone M4A audio, install FFmpeg and run, for example: `ffmpeg -i Angry.m4a -ac 1 -ar 22050 personal/angry.wav`. The converted personal WAVs are in the separate submission ZIP for private reproduction; raw voice files are not tracked in Git by default.

## Method and limitations
180 mean features: 12 chroma, 128 mel (fmax 8 kHz), 40 MFCC. Both datasets are resampled to 22,050 Hz. SVM C=3/gamma=scale; KNN k=5/distance weights; Forest 300 trees/seed42; Logistic C=1/lbfgs/max_iter=10000. StandardScaler is fitted inside a pipeline on training data only. No test-based hyperparameter tuning.

This is a file-level split, not an actor-held-out evaluation. Eight personal samples are exploratory. Unscaled logistic regression emitted a convergence warning at the iteration limit. Tutorial comparisons are contextual because preprocessing, sample count, split and some parameters differ. The supplied tutorial notebook has stored results; those are not claimed as new runs.

## References
Livingstone & Russo (2018), RAVDESS dataset: https://doi.org/10.5281/zenodo.1188976

Tutorial: https://github.com/IAT-ExploringAI-2026/Week3-ClassicML (also see supplied assignment link https://github.com/IAT-ExploringAI-2024/Week3-ClassicML).

librosa feature documentation: https://librosa.org/doc/0.11.0/feature.html

scikit-learn scaling and leakage guidance: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html and https://scikit-learn.org/stable/common_pitfalls.html

Personal dataset: Vanshdeep Singh Sidhu, eight self-recorded emotion clips, October 4, 2026 (Vancouver local date).

