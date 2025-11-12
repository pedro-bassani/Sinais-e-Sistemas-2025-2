import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import pandas as pd

duracao_segundos = 5.0
fs = 500
npontos = int(duracao_segundos * fs)

amplitude1 = 1.0
freq1_hz = 1.0
amplitude2 = 0.5
freq2_hz = 2.0
desvio_ruido = 0.2
faixa_baixa_hz = 0.5
faixa_alta_hz = 40.0
np.random.seed(42)

tempo = np.arange(npontos) / fs
seno1 = amplitude1 * np.sin(2*np.pi*freq1_hz*tempo)
seno2 = amplitude2 * np.sin(2*np.pi*freq2_hz*tempo)
ruido = desvio_ruido * np.random.randn(npontos)
sinal = seno1 + seno2 + ruido

plt.figure(figsize=(10, 4))
plt.plot(tempo, sinal)
plt.title("Sinal sintético de ECG (tempo)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude (u.a.)")
plt.xlim(0, duracao_segundos)
plt.tight_layout()
plt.show()

S = fft(sinal)
frequencias = fftfreq(npontos, d=1/fs)
magnitude_completa = np.abs(S) / npontos
mascara_positiva = frequencias >= 0
freq_positiva = frequencias[mascara_positiva]
S_positiva = S[mascara_positiva]
magnitude_positiva = np.abs(S_positiva) / npontos

plt.figure(figsize=(10, 4))
plt.plot(freq_positiva, magnitude_positiva)
plt.title("Espectro de magnitude |S(f)| — lado único")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (u.a.)")
plt.xlim(0, 60)
plt.tight_layout()
plt.show()

faixa_busca_min = 0.3
faixa_busca_max = 10.0
mascara_banda = (freq_positiva >= faixa_busca_min) & (freq_positiva <= faixa_busca_max)
freq_banda = freq_positiva[mascara_banda]
mag_banda = magnitude_positiva[mascara_banda]
limiar = mag_banda.mean() + 2.0 * mag_banda.std()
indices_pico = np.where(mag_banda >= limiar)[0]
ordem = np.argsort(-mag_banda[indices_pico])
indices_pico = indices_pico[ordem][:5]
freq_picos = freq_banda[indices_pico]
mag_picos = mag_banda[indices_pico]

tabela_picos = pd.DataFrame({"frequencia_Hz": freq_picos, "magnitude": mag_picos})
print("\nPicos detectados (0,3–10 Hz):")
print(tabela_picos.sort_values("frequencia_Hz").to_string(index=False))

plt.figure(figsize=(10, 4))
plt.plot(freq_positiva, magnitude_positiva, label="Espectro")
plt.scatter(freq_picos, mag_picos, s=50, marker="x", label="Picos")
for f, m in zip(freq_picos, mag_picos):
    plt.annotate(f"{f:.2f} Hz", (f, m), xytext=(6, 6), textcoords="offset points", fontsize=8)
plt.title("Picos detectados no espectro")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (u.a.)")
plt.xlim(0, 10)
plt.legend()
plt.tight_layout()
plt.show()

mascara_passa = (np.abs(frequencias) >= faixa_baixa_hz) & (np.abs(frequencias) <= faixa_alta_hz)
H = np.zeros_like(S, dtype=complex)
H[mascara_passa] = 1.0
S_filtrado = S * H
sinal_filtrado = np.real(ifft(S_filtrado))

plt.figure(figsize=(10, 4))
plt.plot(tempo, sinal, label="Original (com ruído)")
plt.plot(tempo, sinal_filtrado, label="Filtrado (0,5–40 Hz)")
plt.title("Comparação no tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude (u.a.)")
plt.xlim(0, duracao_segundos)
plt.legend()
plt.tight_layout()
plt.show()