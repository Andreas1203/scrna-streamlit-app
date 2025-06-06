# scRNA-seq Analysis App (ΤΛ 2025)

Αυτή είναι μια διαδραστική εφαρμογή ανάλυσης δεδομένων scRNA-seq που αναπτύχθηκε για την εργασία του μαθήματος "Τεχνολογία Λογισμικού" (ΤΛ 2025). Η εφαρμογή υλοποιήθηκε με Python και Streamlit και εκτελεί βασικά στάδια ανάλυσης single-cell RNA sequencing.

## Περιεχόμενα

- Προεπεξεργασία δεδομένων (Filtering, Normalization, HVGs)
- UMAP προβολή (2D/3D)
- Clustering (Leiden)
- Διαφορική γονιδιακή έκφραση (DEGs)
- Volcano Plot
- Λήψη αποτελεσμάτων (PNG, CSV, JSON)
- Υποστήριξη UMAP με Harmony integration (batch correction)

---

## 🛠Οδηγίες Εκτέλεσης με Docker

Για να εκτελέσετε την εφαρμογή χωρίς να εγκαταστήσετε χειροκίνητα βιβλιοθήκες, ακολουθήστε τα παρακάτω βήματα:

### 1. Εγκαταστήστε το Docker

https://www.docker.com/

### 2. Κατεβάστε τα αρχεία απο το Github και δημιουργεία φακέλου

- main1.py             # Κύριος κώδικας εφαρμογής Streamlit
- requirements.txt     # Απαιτούμενες βιβλιοθήκες
- Dockerfile           # Οδηγίες για Docker image

! Ο ΦΑΚΕΛΟΣ ΣΑΣ ΠΡΕΠΕΙ ΝΑ ΠΕΡΙΈΧΕΙ ΤΑ ΠΙΟ ΠΑΝΩ ΑΡΧΕΙΑ !

### 3. Για την δημιουργεία της εικόνας του Docker και την εκτέλεση

1. Ανοίξτε το terminal σας
2. Μεταφερθείτε στο φάκελο σας (πχ. cd όνομα-φακέλου)
3. Εκτελέστε αυτην την εντολή docker build -t scrnaseq-app .
4. Στην συνέχεια, εκτελέστε αυτην την εντολή docker run -p 8501:8501 scrnaseq-app
5. Τέλος, ανοίξτε το browser σας και γράψτε http://localhost:8501


