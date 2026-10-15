# Concrete Misclassification Examples

These examples come from the internal validation split. The official unlabelled evaluation set was not used.

## Example 1: Anterior Cervical Discectomy

- Source index: 1747
- Correct label: `Surgery`
- Predicted label: `Neurosurgery`
- Decision source: `clinicalbert_fallback`
- Exact conflicting duplicate in training: True
- Text excerpt: "Anterior cervical discectomy and removal of herniated disk and osteophytes and decompression of spinal cord and bilateral nerve root decompression. Harvesting of autologous bone from the vertebral bodies. Grafting of allograft bone for creation of arthrodesis."

**Interpretation.** This exact input also appears in the training subset with Neurosurgery and Orthopedic labels. The prediction is therefore supported by contradictory annotations, while the procedure itself can reasonably involve several surgical specialties.

## Example 2: Coronary Angiography

- Source index: 1887
- Correct label: `Surgery`
- Predicted label: `Cardiovascular-Pulmonary`
- Decision source: `clinicalbert_fallback`
- Exact conflicting duplicate in training: True
- Text excerpt: "Selective coronary angiography, left heart catheterization with hemodynamics, LV gram with power injection, right femoral artery angiogram, closure of the right femoral artery using 6-French AngioSeal."

**Interpretation.** The document describes a coronary procedure and an exact copy in the training subset is labelled Cardiovascular-Pulmonary. The error reflects ambiguity between the organ-system specialty and the procedural Surgery label.

## Example 3: Abdominal Pain - Consult

- Source index: 507
- Correct label: `Gastroenterology`
- Predicted label: `General Medicine`
- Decision source: `clinicalbert_fallback`
- Exact conflicting duplicate in training: False
- Text excerpt: "The patient presented to the emergency room last evening with approximately 7- to 8-day history of abdominal pain which has been persistent."

**Interpretation.** The short description reports persistent abdominal pain without a specific gastrointestinal diagnosis or procedure. General Medicine is plausible from the available text, showing the difficulty of assigning broad presentations to a single specialty.
