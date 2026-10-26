# NewsGuard Datasets & Attribution

This directory stores the benchmark datasets used for training and evaluating NewsGuard's machine learning models.

---

## 1. Fake or Real News Dataset (Fake News Classification)
- **Source**: George McIntire / Lutz Hamel, University of Rhode Island
- **Repository**: [https://github.com/lutzhamel/fake-news](https://github.com/lutzhamel/fake-news)
- **File**: `data/raw/fake_or_real_news.csv` (30.7 MB)
- **Sample Count**: 6,295 verified articles after preprocessing (3,171 REAL, 3,124 FAKE)
- **Attributes**: `id`, `title`, `text`, `label` (`REAL`, `FAKE`)
- **Origin**:
  - `REAL` articles sourced from reputable journalism publications (Reuters, The New York Times, CNN, BBC, Washington Post).
  - `FAKE` articles collected from known disreputable/disinformation domains flagged during the 2016 US election cycle.
- **License / Usage**: Research & Educational use.
- **Potential Bias & Limitations**:
  - Contains political news primarily from the 2016 US news cycle.
  - Vocabulary distributions reflect that era's political discourse; generalizability to other domains (sports, finance, entertainment) requires ongoing recalibration.

---

## 2. Stop Clickbait Dataset (Clickbait Headline Detection)
- **Authors**: Abhijnan Chakraborty, Bhargavi Paranjape, Sourya Kakarla, and Niloy Ganguly
- **Affiliation**: Indian Institute of Technology (IIT) Kharagpur & Max Planck Institute
- **Paper**: *"Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media"*, Proceedings of the 2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), San Francisco, USA, August 2016.
- **Files**: `data/raw/clickbait_data.gz`, `data/raw/non_clickbait_data.gz`
- **Sample Count**: 32,000 headlines (16,001 Non-Clickbait, 15,999 Clickbait)
- **Origin**:
  - `Clickbait`: Scraped from BuzzFeed, Upworthy, ViralNova, Thatscoop, Scoopwhoop, and ViralStories.
  - `Non-Clickbait`: Sourced from WikiNews, The New York Times, The Guardian, and The Hindu.
- **License / Citation**:
  ```bibtex
  @inproceedings{chakraborty2016stop,
    title={Stop Clickbait: Detecting and preventing clickbaits in online news media},
    author={Chakraborty, Abhijnan and Paranjape, Bhargavi and Kakarla, Sourya and Ganguly, Niloy},
    booktitle={Advances in Social Networks Analysis and Mining (ASONAM), 2016 IEEE/ACM International Conference on},
    pages={9--16},
    year={2016},
    organization={IEEE}
  }
  ```

---

## 3. Headline–Body Mismatch Benchmark
- **Methodology**: Paired Stance & Alignment Protocol.
  - **Positive Class (Consistent)**: 1,400 genuine headline–body pairs from verified journalism where the headline directly introduces the body.
  - **Negative Class (Mismatched)**: 1,400 synthetic swapped pairs where genuine headlines were paired with unrelated news bodies (mimicking bait-and-switch deception).
- **Threshold Calibration**:
  - Evaluated across candidate decision thresholds $\tau \in [0.04, 0.35]$.
  - Calibrated threshold on validation set: $\tau^* = 0.100$.
  - Test set accuracy: $90.58\%$, Macro F1: $0.9055$.
