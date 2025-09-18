# Echo-ABSA-GDM

Aspect-Based Sentiment Analysis (ABSA) + Group Decision Making (GDM) pipeline for measuring **echo chambers** in Indonesian social media conversations.

The pipeline combines:

* **Preprocessing**
* **Domain-independent ABSA** (Gemini or Double Propagation)
* **Interaction graph construction**
* **Community detection** (Louvain / METIS)
* **Consensus calculation (GDM)**

This project was developed as part of my **undergraduate thesis** in
**Teknik Informatika, Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung (ITB).**

---

## 📖 Abstract

Social media has become an important space for Indonesians to interact and form opinions, but communication patterns often create echo chambers that reinforce polarization. However, echo chamber studies in Indonesia remain limited to manual analyses or network-based approaches without considering content sentiment.

This study develops a model to measure echo chambers in Indonesian social media conversations using domain-independent Aspect-Based Sentiment Analysis (ABSA) and consensus metrics within the Group Decision Making (GDM) framework. The proposed pipeline consists of five stages: text preprocessing, domain-independent ABSA, interaction graph construction, community detection, and GDM consensus calculation.

Results show that a zero-shot generative LLM (Gemini-2.5-flash) achieves the highest performance, producing Echo Chamber Ratio (ECR) values consistent with prior expert analyses and statistically significant under null model testing.

---

## 🚀 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/echo-absa-gdm.git
cd echo-absa-gdm
pip install .
```

Requires **Python 3.8+**.

---

## ⚙️ Usage

### CLI Mode

Run the full pipeline from the command line:

```bash
main \
  --community_input data/corpus_final/for_echo_chamber_detection/example.csv \
  --absa_aspect data/corpus_final/for_absa/aspects.csv \
  --workdir pipeline_out \
  --model gemini \
  --model_name gemini-2.5-flash \
  --algo louvain
```

* `--community_input`: CSV with at least `id,text` columns
* `--absa_aspect`: CSV with `aspect,desc` columns
* `--model`: `gemini` (LLM) or `dp` (double propagation)
* `--algo`: Community detection algorithm (`louvain` or `metis`)

Outputs include:

* `absa_community.csv` → merged sentiment + community data
* `consensus.txt` → consensus results
* `community_graph.gml` → interaction graph

---

### GUI Mode

Run the interactive GUI with [NiceGUI](https://nicegui.io/):

```bash
python main_gui.py
```

* Upload input files (community CSV, aspect CSV, optional lexicon CSV for DP)
* Configure model and community detection parameters
* Run the pipeline and view logs + results in the browser

Access at: **[http://localhost:8080/](http://localhost:8080/)**

---

## 📂 Project Structure

```
data/                # Datasets and lexicons
doc/                 # Full thesis document and explanation
src/
  absa/              # ABSA models + metrics
  gdm/               # Group Decision Making consensus
  main/              # Pipeline orchestration
  preprocess/        # Preprocessing scripts
pyproject.toml       # Build + dependency config
main.py              # CLI pipeline entrypoint
main_gui.py          # GUI pipeline with NiceGUI
```

The **`doc/` folder** contains the full thesis document explaining the methodology, experiments, and results in detail.

---

## ✨ Features

* Domain-independent ABSA (Generative LLM or DP-based)
* Automatic community detection from interaction graphs
* Consensus measurement within GDM framework
* CLI + GUI modes
* Extensible for new datasets or models

---

## 👩‍💻 Author

* **Puti Nabilla Aidira**

---

## ℹ️ Disclaimer

This project was created as part of my **undergraduate thesis** in
**Teknik Informatika, Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung (ITB), 2025.**
