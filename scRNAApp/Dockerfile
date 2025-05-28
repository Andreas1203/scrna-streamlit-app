# Χρήση επίσημης ελαφριάς εικόνας Python
FROM python:3.10-slim

# Ορισμός directory εργασίας μέσα στο container
WORKDIR /app

# Αντιγραφή όλων των τοπικών αρχείων στο container
COPY . .

# Ενημέρωση pip και εγκατάσταση εξαρτήσεων
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Άνοιγμα του port που χρησιμοποιεί το Streamlit
EXPOSE 8501

# Εκκίνηση της εφαρμογής Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]
