# Nano-response predictor

**Which bacteria will a chitosan Schiff base nanoparticle affect, and at what dose?**

A small learning project built on the data from my M.Sc. work:

> Soliman MM, Bayoumy AE, Hamed AA, Gabre RM, Abd-Elhakeem MA, Abdelhamid IA, Elgamal AM. Piperazine linked chitosan Schiff base nanoparticles as a novel antibiofilm and antibacterial strategy against clinically relevant pathogens. *Scientific Reports* 16, 24305 (2026). doi:10.1038/s41598-026-63745-z

I am a microbiologist with limited but real programming experience (Python from a bioinformatics diploma, statistics on my own lab data). I built this project to practise a proper data workflow on data I understand well. AI tools helped me write parts of the code; I checked every step and its output.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdelrahman22225-stack/Nano-response-predictor/blob/main/nano_response_predictor.ipynb)

## The questions

1. How does each strain respond as the nanoparticle dose goes up?
2. If we meet a **new** strain, can we predict its nanoparticle response from cheap information (Gram type, biofilm former or not, dose, and the parent compound's zone)?
3. What fraction of the MIC do the nanoparticles need to halve biofilm formation, and where do low doses make biofilm worse?

## Data

| File | What it holds | Source |
|---|---|---|
| `data/inhibition_zones.csv` | Disc-diffusion zones (mm), 9 strains × 2 formulations × 3 doses | Table 1 |
| `data/biofilm_inhibition.csv` | Biofilm inhibition (%), 6 strains × 2 formulations × 0.25, 0.5, 1× MIC | Table 3 |
| `data/strains.csv` | Gram type, Congo red biofilm result, resistance notes | Paper text |

## Method

- Dose-response line per strain: zone = a + b · log10(dose).
- Prediction tested with **leave-one-strain-out**: hide one strain, train on the other eight, predict the hidden one, repeat for all nine.
- Every model is compared with a **baseline** that only knows the dose.
- Biofilm: linear interpolation between the three tested doses to find the dose for 50% inhibition.

## Results

| Model (leave-one-strain-out) | Mean absolute error |
|---|---|
| Baseline (dose only) | 4.60 mm |
| Linear regression | **4.31 mm** |
| Decision tree (depth 2) | 5.28 mm |

- The strongest nanoparticle response was vancomycin-resistant *E. faecalis* ATCC 51299 (28.3 mm at 10 mg/mL); the weakest was *P. aeruginosa* ATCC 27853 (7.7 mm).
- With only nine strains, the features improve on the dose-only baseline only slightly. That is the honest answer: the data are too small to predict a new strain reliably.
- In the linear model, being a biofilm former (Congo red positive) went with a larger nanoparticle zone. This is a hypothesis to test, not a finding.
- The nanoparticles reached 50% biofilm inhibition at 0.25–0.69× their MIC, and the parent compound at 0.30–0.96× its MIC. These are fractions of each formulation's own MIC, and the nanoparticle MICs are higher, so they are not equal absolute doses.
- At 0.25× MIC, three strains (*S. aureus* 29213 and 1026, *E. coli* 35218) made **more** biofilm than untreated controls, so the dose must stay at or above the MIC.

## Limitations

- Nine strains, three doses, published means only (no replicate-level data).
- A zone of 0 mm means "no zone seen", not a measured value, and a straight line treats it as one.
- Crystal violet measures biofilm biomass, not viability, and was not normalised to planktonic growth.

## Next steps

- Add clinical isolates and measured features (membrane charge, efflux, capsule) before any real prediction.
- Use replicate-level data and fit proper dose-response (Hill) curves.

## Run it

**Google Colab (no installation):** click the badge above, then Runtime → Run all.

**On your computer:**

```bash
pip install -r requirements.txt
python analysis.py
```

Figures are saved in `figures/`.

## Use it with your own data

1. Keep the column names exactly as they are in the three CSV files.
2. Add one row per strain to `data/strains.csv` (Gram type 1 or 0, Congo red biofilm result 1 or 0).
3. Add that strain's zones to `data/inhibition_zones.csv` (both formulations, the same three doses) and, if you have them, its biofilm values to `data/biofilm_inhibition.csv`.
4. Run the notebook or `analysis.py` again. The leave-one-strain-out test automatically includes the new strains.

To test a different compound, replace the rows with your own measurements in the same format. More strains make the prediction test more meaningful.

## Licence

Code: MIT. Data reproduced from the open-access article above (CC BY 4.0).
