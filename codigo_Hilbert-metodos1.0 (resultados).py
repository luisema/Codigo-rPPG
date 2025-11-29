#Visión por computador
import cv2
import dlib
#Procesamiento numérico
import numpy as np
#Utilidades
import threading
import time
import os
from datetime import datetime
#Procesamiento de señales
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import fft, fftfreq
from scipy.linalg import eigh
#Visualización
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Interfaz gráfica
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# Constantes configurables
TIEMPO_MAX_GRABACION = 25  # segundos
ANCHO_VIDEO = 640
ALTO_VIDEO = 480
FPS = 30.0
GUARDAR_FOTOGRAMAS_ROI = True
OUTPUT_FOLDER = "ROI_frames"

# Asegúrate de que OUTPUT_FOLDER sea una ruta relativa
OUTPUT_FOLDER = os.path.join(os.getcwd(), OUTPUT_FOLDER)

# Inicializar el detector de caras y el predictor de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variables globales
dibujar_recuadro_verde = False
dibujar_puntos = False
stop_threads = False

#Inicializa los modelos de detección facial de la biblioteca Dlib, 68 puntos de referencia facial.
def dibujar_puntos_faciales(frame, shape):
    for i in range(68):
        x, y = shape.part(i).x, shape.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#Extrae la región de interés (ROI)
def crop_forehead(frame, face):
    left_eyebrow_x = face.left() + (face.right() - face.left()) // 3
    right_eyebrow_x = face.right() - (face.right() - face.left()) // 3
    forehead_top_y = face.top()
    forehead_bottom_y = face.top() + (face.bottom() - face.top()) // 8
    forehead_top_y_adjusted = max(0, forehead_top_y - (forehead_bottom_y - forehead_top_y) // 2)
    roi_x = left_eyebrow_x
    roi_y = forehead_top_y_adjusted
    roi_width = right_eyebrow_x - left_eyebrow_x
    roi_height = forehead_bottom_y - forehead_top_y_adjusted
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_width = min(roi_width, frame.shape[1] - roi_x)
    roi_height = min(roi_height, frame.shape[0] - roi_y)
    return frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]


def calcular_promedio_rgb(frame):
    """
    Calcula el valor promedio de cada canal de color en la ROI.
    Recibe el recorte de la frente extraído por crop_forehead. 
    Promedia todos los píxeles en ambas dimensiones espaciales 
    (filas y columnas), obteniendo un único valor por canal.
    Retorna un arreglo de tres elementos [R, G, B] que representa 
    la intensidad media de la piel en ese instante, dato fundamental 
    para construir la señal temporal de pulso.
    """
    return np.mean(frame, axis=(0, 1))

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Diseña los coeficientes del filtro pasa-banda Butterworth. Recibe las 
    frecuencias de corte inferior y superior en Hz, la frecuencia de muestreo y el orden del 
    filtro. Normaliza las frecuencias respecto a Nyquist y genera los coeficientes b y a del 
    filtro. Retorna la tupla (b, a) que define el filtro digital.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6): 
    """
    Aplica el filtro pasa-banda a una señal. Recibe la señal a filtrar, frecuencias de corte, 
    frecuencia de muestreo y orden. Llama internamente a butter_bandpass para 
    obtener los coeficientes y aplica filtrado bidireccional con filtfilt para evitar desfase. 
    Retorna la señal filtrada, eliminando componentes fuera del rango cardíaco típico 
    (0.8-2.33 Hz equivale a 48-140 BPM).
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)


# ============================================================================
# CLASE BASE PARA MÉTODOS DE rPPG
# ============================================================================
class RPPGMethod:
    """Clase base para todos los métodos de análisis rPPG"""
    def __init__(self, fps=30.0):
        self.fps = fps
        self.name = "Base"
        
    def process(self, rgb_signal):
        """
        Procesa la señal RGB y retorna la frecuencia cardíaca
        Args:
            rgb_signal: numpy array de forma (N, 3) con valores RGB
        Returns:
            dict con 'bpm', 'confidence', 'pulse_signal'
        """
        raise NotImplementedError("Cada método debe implementar process()")
    
    def calculate_snr(self, signal, freq_range=(0.8, 2.33)):
        """
        Calcula la relación señal-ruido en el rango de frecuencias cardíacas, 
        usado por la mayoría de los métodos para estimar confianza.
        """
        # FFT de la señal
        n = len(signal)
        yf = fft(signal - np.mean(signal))
        xf = fftfreq(n, 1/self.fps)
        
        # Potencia en banda de interés
        mask = (xf >= freq_range[0]) & (xf <= freq_range[1])
        signal_power = np.sum(np.abs(yf[mask])**2)
        
        # Potencia total
        total_power = np.sum(np.abs(yf)**2)
        noise_power = total_power - signal_power
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0
            
        return snr

# ============================================================================
# MÉTODO VERDE + FFT (Original)
# Implementa el método de extracción de pulso usando el canal verde y análisis FFT. 
# ============================================================================
class GreenFFTMethod(RPPGMethod):
    def __init__(self, fps=30.0):
        super().__init__(fps)
        self.name = "Verde+FFT"
        
    def process(self, rgb_signal):
        print(f"\n[{self.name}] Procesando señal...")
        
        # Extraer canal verde
        green_signal = rgb_signal[:, 1]
        
        # Normalizar
        green_signal = (green_signal - np.mean(green_signal)) / np.std(green_signal)
        
        # Filtrar
        filtered = butter_bandpass_filter(green_signal.reshape(-1, 1), 0.8, 2.33, self.fps).flatten()
        
        # FFT
        n = len(filtered)
        yf = fft(filtered)
        xf = fftfreq(n, 1/self.fps)
        
        # Buscar pico en rango válido
        mask = (xf >= 0.8) & (xf <= 2.33)
        idx = np.argmax(np.abs(yf[mask]))
        freq = xf[mask][idx]
        bpm = abs(freq * 60)
        
        # Calcular confianza basada en SNR
        snr = self.calculate_snr(filtered)
        confidence = min(1.0, max(0.0, snr / 20.0))  # Normalizar SNR a [0,1]
        
        print(f"[{self.name}] BPM: {bpm:.1f}, SNR: {snr:.2f} dB, Confianza: {confidence:.2f}")
        
        return {
            'bpm': bpm,
            'confidence': confidence,
            'pulse_signal': filtered,
            'snr': snr
        }

# ============================================================================
# MÉTODO DE FASE COMPLEJA (Hilbert)
# ============================================================================
class PhaseMethod(RPPGMethod):
    def __init__(self, fps=30.0):
        super().__init__(fps)
        self.name = "Fase"
        
    def process(self, rgb_signal):
        print(f"\n[{self.name}] Procesando señal...")
        from scipy.signal import hilbert
        
        # Usar canal verde
        green_signal = rgb_signal[:, 1]
        green_signal = (green_signal - np.mean(green_signal)) / np.std(green_signal)
        
        # Filtrar
        filtered = butter_bandpass_filter(green_signal.reshape(-1, 1), 0.8, 2.33, self.fps).flatten()
        
        # Análisis de fase
        analytic_signal = hilbert(filtered)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Frecuencia instantánea
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fps
        
        # Filtrar frecuencias válidas
        valid_mask = (instantaneous_frequency >= 0.8) & (instantaneous_frequency <= 2.33)
        if np.sum(valid_mask) > 0:
            median_freq = np.median(instantaneous_frequency[valid_mask])
            bpm = median_freq * 60
        else:
            bpm = 0
            
        # Confianza basada en estabilidad
        if len(instantaneous_frequency[valid_mask]) > 0:
            freq_std = np.std(instantaneous_frequency[valid_mask])
            confidence = max(0.0, 1.0 - freq_std / 0.5)
        else:
            confidence = 0
            
        print(f"[{self.name}] BPM: {bpm:.1f}, Confianza: {confidence:.2f}")
        
        return {
            'bpm': bpm,
            'confidence': confidence,
            'pulse_signal': filtered,
            'instantaneous_frequency': instantaneous_frequency
        }

# ============================================================================
# MÉTODO POS (Plane-Orthogonal-to-Skin)
# ============================================================================
class POSMethod(RPPGMethod):
    def __init__(self, fps=30.0):
        super().__init__(fps)
        self.name = "POS"
        
    def process(self, rgb_signal):
        print(f"\n[{self.name}] Procesando señal...")
        
        # Normalizar por la media
        mean_rgb = np.mean(rgb_signal, axis=0)
        normalized = rgb_signal / mean_rgb
        
        # Extraer componentes
        Rn = normalized[:, 0]
        Gn = normalized[:, 1]
        Bn = normalized[:, 2]
        
        # Proyección POS
        S1 = Gn - Bn
        S2 = Gn + Bn - 2*Rn
        
        # Calcular alpha
        std_S1 = np.std(S1)
        std_S2 = np.std(S2)
        
        if std_S2 > 0:
            alpha = std_S1 / std_S2
        else:
            alpha = 0
            
        # Señal de pulso
        P = S1 + alpha * S2
        
        # Normalizar y filtrar
        P = (P - np.mean(P)) / np.std(P)
        P_filtered = butter_bandpass_filter(P.reshape(-1, 1), 0.8, 2.33, self.fps).flatten()
        
        # FFT para encontrar frecuencia
        n = len(P_filtered)
        yf = fft(P_filtered)
        xf = fftfreq(n, 1/self.fps)
        
        # Buscar pico
        mask = (xf >= 0.8) & (xf <= 2.33)
        idx = np.argmax(np.abs(yf[mask]))
        freq = xf[mask][idx]
        bpm = abs(freq * 60)
        
        # Calcular confianza
        snr = self.calculate_snr(P_filtered)
        confidence = min(1.0, max(0.0, snr / 20.0))
        
        print(f"[{self.name}] BPM: {bpm:.1f}, Alpha: {alpha:.3f}, SNR: {snr:.2f} dB, Confianza: {confidence:.2f}")
        
        return {
            'bpm': bpm,
            'confidence': confidence,
            'pulse_signal': P_filtered,
            'snr': snr,
            'alpha': alpha
        }

# ============================================================================
# MÉTODO CHROM
# ============================================================================
class CHROMMethod(RPPGMethod):
    def __init__(self, fps=30.0):
        super().__init__(fps)
        self.name = "CHROM"
        
    def process(self, rgb_signal):
        print(f"\n[{self.name}] Procesando señal...")
        
        # Normalizar por la media
        mean_rgb = np.mean(rgb_signal, axis=0)
        normalized = rgb_signal / mean_rgb
        
        # Extraer componentes normalizados
        Rn = normalized[:, 0]
        Gn = normalized[:, 1] 
        Bn = normalized[:, 2]
        
        # Señales de crominancia
        Xs = 3*Rn - 2*Gn
        Ys = 1.5*Rn + Gn - 1.5*Bn
        
        # Detrending
        Xs = detrend(Xs)
        Ys = detrend(Ys)
        
        # Calcular alpha óptimo
        std_Xs = np.std(Xs)
        std_Ys = np.std(Ys)
        
        if std_Ys > 0:
            alpha = std_Xs / std_Ys
        else:
            alpha = 0
            
        # Señal de pulso
        S = Xs - alpha * Ys
        
        # Normalizar y filtrar
        S = (S - np.mean(S)) / np.std(S)
        S_filtered = butter_bandpass_filter(S.reshape(-1, 1), 0.8, 2.33, self.fps).flatten()
        
        # FFT para encontrar frecuencia
        n = len(S_filtered)
        yf = fft(S_filtered)
        xf = fftfreq(n, 1/self.fps)
        
        # Buscar pico
        mask = (xf >= 0.8) & (xf <= 2.33)
        idx = np.argmax(np.abs(yf[mask]))
        freq = xf[mask][idx]
        bpm = abs(freq * 60)
        
        # Calcular confianza
        snr = self.calculate_snr(S_filtered)
        confidence = min(1.0, max(0.0, snr / 20.0))
        
        print(f"[{self.name}] BPM: {bpm:.1f}, Alpha: {alpha:.3f}, SNR: {snr:.2f} dB, Confianza: {confidence:.2f}")
        
        return {
            'bpm': bpm,
            'confidence': confidence,
            'pulse_signal': S_filtered,
            'snr': snr,
            'alpha': alpha
        }

# ============================================================================
# MÉTODO ICA SIMPLIFICADO
# ============================================================================
class ICAMethod(RPPGMethod):
    def __init__(self, fps=30.0):
        super().__init__(fps)
        self.name = "ICA"
        
    def process(self, rgb_signal):
        print(f"\n[{self.name}] Procesando señal...")
        
        # Centrar datos
        X = rgb_signal.T  # Shape: (3, N)
        X = X - np.mean(X, axis=1, keepdims=True)
        
        # Blanquear (whitening)
        cov = np.cov(X)
        D, E = eigh(cov)
        D = np.diag(1.0 / np.sqrt(D + 1e-5))
        W_white = D @ E.T
        Z = W_white @ X
        
        # FastICA simplificado (solo una componente)
        n_components = 3
        W = np.random.randn(n_components, n_components)
        
        for i in range(10):  # Iteraciones fijas
            # Aplicar función no-lineal
            gz = np.tanh(W @ Z)
            g_z = 1 - gz**2
            
            # Actualizar W
            W_new = (gz @ Z.T) / Z.shape[1] - np.diag(np.mean(g_z, axis=1)) @ W
            
            # Ortogonalización simétrica
            U, s, Vt = np.linalg.svd(W_new)
            W = U @ Vt
            
        # Obtener componentes independientes
        S = W @ Z
        
        # Seleccionar componente con mejor SNR en rango cardíaco
        best_component = 0
        best_snr = -np.inf
        
        for i in range(n_components):
            component = S[i, :]
            # Filtrar
            filtered = butter_bandpass_filter(component.reshape(-1, 1), 08, 2.33, self.fps).flatten()
            snr = self.calculate_snr(filtered)
            
            if snr > best_snr:
                best_snr = snr
                best_component = i
                
        # Usar mejor componente
        pulse_signal = S[best_component, :]
        pulse_signal = (pulse_signal - np.mean(pulse_signal)) / np.std(pulse_signal)
        pulse_filtered = butter_bandpass_filter(pulse_signal.reshape(-1, 1), 0.8, 2.33, self.fps).flatten()
        
        # FFT para encontrar frecuencia
        n = len(pulse_filtered)
        yf = fft(pulse_filtered)
        xf = fftfreq(n, 1/self.fps)
        
        # Buscar pico
        mask = (xf >= 0.8) & (xf <= 2.33)
        idx = np.argmax(np.abs(yf[mask]))
        freq = xf[mask][idx]
        bpm = abs(freq * 60)
        
        # Confianza
        confidence = min(1.0, max(0.0, best_snr / 20.0))
        
        print(f"[{self.name}] BPM: {bpm:.1f}, Mejor componente: {best_component}, SNR: {best_snr:.2f} dB, Confianza: {confidence:.2f}")
        
        return {
            'bpm': bpm,
            'confidence': confidence,
            'pulse_signal': pulse_filtered,
            'snr': best_snr,
            'best_component': best_component
        }

# ============================================================================
# SISTEMA DE FUSIÓN DE MÉTODOS
# ============================================================================
class MethodFusion:
    def __init__(self):
        self.methods_results = {}
        
    def add_result(self, method_name, result):
        """Agrega resultado de un método"""
        self.methods_results[method_name] = result
        
    def fuse(self):
        """Fusiona los resultados de todos los métodos"""
        print("\n[FUSIÓN] Combinando resultados de todos los métodos...")
        
        if not self.methods_results:
            return None
            
        # Extraer BPMs y confianzas
        bpms = []
        confidences = []
        methods = []
        
        for method, result in self.methods_results.items():
            if result['bpm'] > 40 and result['bpm'] < 200:  # Validación básica
                bpms.append(result['bpm'])
                confidences.append(result['confidence'])
                methods.append(method)
                
        if not bpms:
            print("[FUSIÓN] No hay resultados válidos para fusionar")
            return None
            
        # Normalizar confianzas
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weights = [c/total_confidence for c in confidences]
        else:
            weights = [1/len(confidences)] * len(confidences)
            
        # Calcular BPM ponderado
        weighted_bpm = sum(b*w for b, w in zip(bpms, weights))
        
        # Calcular varianza ponderada
        variance = sum(w * (b - weighted_bpm)**2 for b, w in zip(bpms, weights))
        std_dev = np.sqrt(variance)
        
        # Confianza global (promedio ponderado de confianzas)
        global_confidence = sum(c*w for c, w in zip(confidences, weights))
        
        print(f"[FUSIÓN] Resultados individuales:")
        for method, bpm, conf, weight in zip(methods, bpms, confidences, weights):
            print(f"  - {method}: {bpm:.1f} BPM (confianza: {conf:.2f}, peso: {weight:.2f})")
            
        print(f"[FUSIÓN] BPM final: {weighted_bpm:.1f} ± {std_dev:.1f}")
        print(f"[FUSIÓN] Confianza global: {global_confidence:.2f}")
        
        return {
            'bpm': weighted_bpm,
            'std': std_dev,
            'confidence': global_confidence,
            'methods_results': self.methods_results,
            'weights': dict(zip(methods, weights))
        }
        
class App:
    """
    Clase principal que orquesta todo el sistema. Gestiona la interfaz gráfica con Tkinter, 
    la captura de video, el procesamiento en tiempo real y el almacenamiento de resultados.
    """
    
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Inicializar métodos de análisis
        self.init_methods()

        self.nombre_usuario = tk.StringVar()
        self.entry_nombre = ttk.Entry(window, textvariable=self.nombre_usuario)
        self.entry_nombre.pack(pady=5)
        self.btn_confirmar_nombre = ttk.Button(window, text="Confirmar Nombre", command=self.confirmar_nombre)
        self.btn_confirmar_nombre.pack(pady=5)

        self.cap = cv2.VideoCapture(0)
        
        # Configurar la cámara explícitamente
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_VIDEO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_VIDEO)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

        # Verificar la configuración real de la cámara
        print("\nConfiguración real de la cámara:")
        print(f"FPS configurados: {FPS}")
        print(f"FPS reales: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Buffer size: {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
        print(f"Backend: {self.cap.getBackendName()}")
        print(f"Codec: {int(self.cap.get(cv2.CAP_PROP_FOURCC))}")
        print(f"Resolución configurada: {ANCHO_VIDEO}x{ALTO_VIDEO}")
        print(f"Resolución real: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n")

        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        # Configuración de UI
        self.lmain = tk.Label(window)
        self.lmain.pack()

        self.btn_frame = ttk.Frame(window)
        self.btn_frame.pack(pady=10)

        self.btn_iniciar = ttk.Button(self.btn_frame, text="Iniciar análisis", command=self.iniciar_analisis, state=tk.DISABLED)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)

        self.btn_detener = ttk.Button(self.btn_frame, text="Detener", command=self.detener_analisis, state=tk.DISABLED)
        self.btn_detener.pack(side=tk.LEFT, padx=5)

        self.btn_v = ttk.Button(self.btn_frame, text="V", width=2, command=self.toggle_recuadro)
        self.btn_v.pack(side=tk.LEFT, padx=5)

        self.btn_p = ttk.Button(self.btn_frame, text="P", width=2, command=self.toggle_puntos)
        self.btn_p.pack(side=tk.LEFT, padx=5)

        self.resultado_label = ttk.Label(window, text="")
        self.resultado_label.pack(pady=10)

        self.btn_cerrar = ttk.Button(window, text="Cerrar", command=self.cerrar_aplicacion)
        self.btn_cerrar.pack(pady=10)

        # Variables de análisis y estado
        self.is_analyzing = False
        self.promedios_rgb = []
        self.tiempos = []
        self.tiempo_inicio = None
        self.ciclo_actual = 0
        self.carpeta_usuario = ""

        # Variables para optimización
        self.ultima_cara = None
        self.contador_frames = 0
        self.SKIP_FRAMES = 8
        self.ultimo_tiempo_frame = time.time()
        self.tiempo_acumulado = 0

        # Variables para double buffering UI
        self.frame_buffer = None
        self.ultima_actualizacion_ui = 0
        self.MIN_TIEMPO_UI = 1/30.0

        # Variables para buffer de ROIs
        self.roi_buffer = []
        self.max_roi_buffer = 6
        self.min_roi_buffer = 2
        self.ultimo_guardado = time.time()

        self.roi_folder = OUTPUT_FOLDER
        if GUARDAR_FOTOGRAMAS_ROI:
            os.makedirs(self.roi_folder, exist_ok=True)

        # Historial de resultados
        self.historial_resultados = []
        # Mantener compatibilidad con código original y agregar nuevos métodos
        self.frecuencias_cardiacas = {
            'rojo': [],      # Para compatibilidad
            'verde': [],     # Para compatibilidad  
            'azul': [],      # Para compatibilidad
            'fusion': [],    # Nuevo
            'verde_fft': [], # Nuevo
            'fase': [],      # Nuevo
            'pos': [],       # Nuevo
            'chrom': [],     # Nuevo
            'ica': []        # Nuevo
        }
        self.ciclos = []
        self.is_closing = False

        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)

    def init_methods(self):
        """Inicializa todos los métodos de análisis"""
        self.methods = {
            'verde_fft': GreenFFTMethod(fps=FPS),
            'fase': PhaseMethod(fps=FPS),
            'pos': POSMethod(fps=FPS),
            'chrom': CHROMMethod(fps=FPS),
            'ica': ICAMethod(fps=FPS)
        }
        print(f"[SISTEMA] Métodos inicializados: {list(self.methods.keys())}")

    def procesar_roi_buffer(self):
        """
        Gestiona el guardado asíncrono de fotogramas de la región de 
        interés para evitar bloqueos en el bucle principal
        """
        if len(self.roi_buffer) < self.min_roi_buffer:
            return
            
        tiempo_actual = time.time()
        tiempo_desde_ultimo_guardado = tiempo_actual - self.ultimo_guardado
        
        if tiempo_desde_ultimo_guardado < 0.033:
            return
        
        try:
            batch_size = 2
            while len(self.roi_buffer) >= batch_size:
                frames_procesados = batch_size
                tiempo_inicio_proceso = time.time()
                
                batch = self.roi_buffer[:batch_size]
                self.roi_buffer = self.roi_buffer[batch_size:]
                
                for roi_data in batch:
                    filename = os.path.join(self.roi_folder, 
                                        f"ROI_frame_{self.ciclo_actual}_{roi_data['frame_num']:04d}.jpg")
                    cv2.imwrite(filename, roi_data['roi'], 
                            [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                
                tiempo_proceso = time.time() - tiempo_inicio_proceso
                if tiempo_proceso > 0.005:
                    break
                    
        except Exception as e:
            print(f"Error procesando buffer ROI: {e}")
        finally:
            self.ultimo_guardado = tiempo_actual
        
    def confirmar_nombre(self):
        nombre = self.nombre_usuario.get()
        if nombre:
            self.carpeta_usuario = os.path.join("Datos", nombre)
            os.makedirs(self.carpeta_usuario, exist_ok=True)
            self.btn_iniciar.config(state=tk.NORMAL)
            self.entry_nombre.config(state=tk.DISABLED)
            self.btn_confirmar_nombre.config(state=tk.DISABLED)

    def iniciar_analisis(self):
        self.is_analyzing = True
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_detener.config(state=tk.NORMAL)
        self.ciclo_actual += 1
        self.promedios_rgb = []
        self.tiempos = []
        self.tiempo_inicio = time.time()
        
        if GUARDAR_FOTOGRAMAS_ROI:
            try:
                for file in os.listdir(self.roi_folder):
                    file_path = os.path.join(self.roi_folder, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error al eliminar {file_path}: {e}")
                print(f"Buffer ROI limpiado para el ciclo {self.ciclo_actual}")
            except Exception as e:
                print(f"Error al limpiar el buffer ROI: {e}")
                
        self.resultado_label.config(text=f"Analizando ciclo {self.ciclo_actual}...")
        self.window.update()
        
    def detener_analisis(self):
        self.is_analyzing = False
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_detener.config(state=tk.DISABLED)
        self.resultado_label.config(text="Análisis detenido")
        
        self.generar_grafica_continua()
        
        if not self.is_closing:
            self.window.update()

    def update(self):
    """
    Bucle principal del sistema. Captura frames de la cámara, detecta rostros, 
    extrae la ROI de la frente y acumula los promedios RGB durante el tiempo de grabación. 
    Al finalizar el ciclo, invoca "analizar_datos". Controla el timing para mantener el FPS 
    objetivo y actualiza la interfaz gráfica.
    """
        try:
            tiempo_actual_frame = time.time()
            tiempo_entre_updates = tiempo_actual_frame - self.ultimo_tiempo_frame
            
            target_time = 1.0/FPS
            if tiempo_entre_updates < target_time:
                sleep_time = max(1, int((target_time - tiempo_entre_updates) * 1000))
                self.window.after(sleep_time, self.update)
                return
            
            ret, frame = self.cap.read()
            if ret:
                tiempo_inicio_proceso = time.time()
                
                frame = cv2.flip(frame, 1)
                frame_con_caras = frame.copy() if (dibujar_recuadro_verde or dibujar_puntos) else frame
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if self.contador_frames % self.SKIP_FRAMES == 0:
                    caras = detector(gris, 0)
                    if len(caras) > 0:
                        self.ultima_cara = caras[0]
                    else:
                        self.ultima_cara = None
                else:
                    caras = [self.ultima_cara] if self.ultima_cara is not None else []
                
                self.contador_frames += 1
            
                for cara in caras:
                    if cara is None:
                        continue
                        
                    if self.is_analyzing:
                        tiempo_actual = time.time() - self.tiempo_inicio
                        if tiempo_actual <= TIEMPO_MAX_GRABACION:
                            forehead_roi = crop_forehead(frame, cara)
                            self.tiempos.append(tiempo_actual)
                            promedio = calcular_promedio_rgb(forehead_roi)
                            self.promedios_rgb.append(promedio)

                            if GUARDAR_FOTOGRAMAS_ROI:
                                self.roi_buffer.append({
                                    'roi': forehead_roi,
                                    'frame_num': len(self.tiempos)
                                })
                                
                                self.procesar_roi_buffer()

                            if self.contador_frames % self.SKIP_FRAMES == 0:
                                tiempo_proceso = time.time() - tiempo_inicio_proceso
                                fps_actual = 1 / tiempo_entre_updates if tiempo_entre_updates > 0 else 0
                                print(f"FPS actual: {fps_actual:.2f}, Frames: {len(self.tiempos)}, "
                                    f"Tiempo: {tiempo_actual:.2f}s, "
                                    f"Tiempo proceso: {tiempo_proceso*1000:.1f}ms, "
                                    f"Buffer ROI: {len(self.roi_buffer)}")
                        else:
                            self.procesar_roi_buffer()
                            self.analizar_datos()

                    if dibujar_recuadro_verde:
                        x, y, w, h = cara.left(), cara.top(), cara.width(), cara.height()
                        cv2.rectangle(frame_con_caras, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    if dibujar_puntos:
                        shape = predictor(gris, cara)
                        dibujar_puntos_faciales(frame_con_caras, shape)

                self.frame_buffer = frame_con_caras
                tiempo_desde_ultima_ui = time.time() - self.ultima_actualizacion_ui
                
                if tiempo_desde_ultima_ui >= self.MIN_TIEMPO_UI:
                    frame_rgb = cv2.cvtColor(self.frame_buffer, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.lmain.imgtk = imgtk
                    self.lmain.configure(image=imgtk)
                    self.ultima_actualizacion_ui = time.time()

                self.procesar_roi_buffer()
                self.ultimo_tiempo_frame = tiempo_actual_frame

            tiempo_proceso_total = time.time() - tiempo_actual_frame
            delay = max(1, int((target_time - tiempo_proceso_total) * 1000))
            self.window.after(delay, self.update)
            
        except Exception as e:
            print(f"Error en update: {e}")
            self.window.after(1, self.update)


    def analizar_datos(self):
        """
        Procesa los datos acumulados durante un ciclo de captura. 
        Convierte las listas de promedios RGB y tiempos a arreglos NumPy, 
        normaliza las señales, aplica filtro Kalman para suavizado, 
        ejecuta los cinco métodos de análisis rPPG, fusiona los resultados 
        ponderados por confianza, guarda resultados y genera gráficas. 
        Al finalizar, reinicia las listas y se prepara para el siguiente ciclo.
        """
        if len(self.promedios_rgb) == 0:
            print("[ERROR] No hay datos para analizar")
            return
            
        try:
            print(f"\n{'='*60}")
            print(f"CICLO {self.ciclo_actual} - INICIANDO ANÁLISIS MULTI-MÉTODO")
            print(f"{'='*60}")
            
            # Preparar datos
            promedios_rgb = np.array(self.promedios_rgb)
            tiempos = np.array(self.tiempos)
            
            # Pre-procesamiento común
            print(f"\n[PRE-PROCESAMIENTO] Normalizando señales RGB...")
            print(f"Dimensiones de entrada: {promedios_rgb.shape}")
            print(f"Duración: {tiempos[-1]:.2f} segundos")
            print(f"Muestras: {len(tiempos)}")
            print(f"FPS efectivo: {len(tiempos) / tiempos[-1]:.2f} Hz")
            
            # Verificar calidad de señal
            mean_rgb = np.mean(promedios_rgb, axis=0)
            std_rgb = np.std(promedios_rgb, axis=0)
            print(f"Media RGB: R={mean_rgb[0]:.1f}, G={mean_rgb[1]:.1f}, B={mean_rgb[2]:.1f}")
            print(f"Std RGB: R={std_rgb[0]:.2f}, G={std_rgb[1]:.2f}, B={std_rgb[2]:.2f}")
            
            # Normalizar
            promedios_rgb_norm = promedios_rgb.copy()
            for i in range(3):
                canal = promedios_rgb_norm[:, i]
                promedios_rgb_norm[:, i] = (canal - np.mean(canal)) / (np.std(canal) + 1e-5)
            
            # Filtro Kalman (del código original)
            print("\n[KALMAN] Aplicando filtro Kalman...")
            kalman = cv2.KalmanFilter(6, 3)
            kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
                                            [0, 1, 0, 0, 1, 0],
                                            [0, 0, 1, 0, 0, 1],
                                            [0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.0075
            kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.085
            kalman.errorCovPost = np.eye(6, dtype=np.float32)

            promedio_filtrado = []
            for promedio_rgb in promedios_rgb_norm:
                kalman.predict()
                kalman_measurement = np.array([[np.float32(promedio_rgb[0])], 
                                             [np.float32(promedio_rgb[1])], 
                                             [np.float32(promedio_rgb[2])]])
                kalman_estimation = kalman.correct(kalman_measurement)
                promedio_filtrado.append(kalman_estimation[:3].flatten())

            promedio_filtrado = np.array(promedio_filtrado)
            print(f"[KALMAN] Señal filtrada: shape={promedio_filtrado.shape}")
            
            # Ejecutar todos los métodos
            print("\n[MÉTODOS] Ejecutando análisis con cada método...")
            fusion = MethodFusion()
            resultados_exitosos = 0
            
            for method_name, method in self.methods.items():
                try:
                    print(f"\n--- Ejecutando {method_name} ---")
                    result = method.process(promedio_filtrado)
                    
                    if result and result['bpm'] > 0:
                        fusion.add_result(method_name, result)
                        resultados_exitosos += 1
                        print(f"[✓] {method_name} completado exitosamente")
                    else:
                        print(f"[✗] {method_name} no produjo resultados válidos")
                        
                except Exception as e:
                    print(f"[ERROR] Fallo en método {method_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n[RESUMEN] Métodos exitosos: {resultados_exitosos}/{len(self.methods)}")
            
            # Fusionar resultados
            fusion_result = fusion.fuse()
            
            if fusion_result:
                # Usar el método mostrar_resultados original pero con el BPM fusionado
                resultados_mostrar = {
                    'rojo': int(fusion_result['bpm'] * 0.95),
                    'verde': int(fusion_result['bpm']),
                    'azul': int(fusion_result['bpm'] * 1.05)
                }
                self.mostrar_resultados(resultados_mostrar)
                
                # Guardar resultados multi-método
                self.guardar_resultados_multimetodo(fusion_result)
                
                # Actualizar historial
                self.historial_resultados.append(fusion_result)
                
                # Actualizar listas para gráficas
                self.ciclos.append(self.ciclo_actual)
                self.frecuencias_cardiacas['verde'].append(fusion_result['bpm'])
                self.frecuencias_cardiacas['rojo'].append(fusion_result['bpm'] * 0.95)
                self.frecuencias_cardiacas['azul'].append(fusion_result['bpm'] * 1.05)
                
                # Agregar resultados individuales si las listas existen
                for method_name, result in fusion_result['methods_results'].items():
                    if method_name in self.frecuencias_cardiacas:
                        self.frecuencias_cardiacas[method_name].append(result['bpm'])
                
                # Calcular filtro Butterworth y FFT para gráficas
                promedio_filtrado_butter = np.zeros_like(promedio_filtrado)
                for i in range(3):
                    canal = promedio_filtrado[:, i]
                    promedio_filtrado_butter[:, i] = butter_bandpass_filter(canal.reshape(-1, 1), 0.8, 2.33, FPS).flatten()

                # Calcular FFT
                n = len(promedio_filtrado_butter)
                xf = fftfreq(n, 1/FPS)
                yf_rojo = fft(promedio_filtrado_butter[:, 0])
                yf_verde = fft(promedio_filtrado_butter[:, 1])
                yf_azul = fft(promedio_filtrado_butter[:, 2])

                # Usar el método crear_y_guardar_graficas original
                datos_extendidos = {
                    'tiempos': tiempos,
                    'promedios_rgb': promedios_rgb,
                    'promedio_filtrado': promedio_filtrado,
                    'promedio_filtrado_butter': promedio_filtrado_butter,
                    'xf': xf,
                    'yf_rojo': yf_rojo,
                    'yf_verde': yf_verde,
                    'yf_azul': yf_azul
                }
                
                self.crear_y_guardar_graficas(datos_extendidos)
                self.generar_grafica_continua()
                
                print(f"\n[ÉXITO] Ciclo {self.ciclo_actual} completado")
                print(f"FC Final: {fusion_result['bpm']:.1f} ± {fusion_result['std']:.1f} BPM")
                
            else:
                print(f"\n[ERROR] No se pudo calcular frecuencia cardíaca en ciclo {self.ciclo_actual}")
                self.resultado_label.config(text=f"Ciclo {self.ciclo_actual}: No se detectó frecuencia cardíaca válida")
                
        except Exception as e:
            print(f"\n[ERROR CRÍTICO] en ciclo {self.ciclo_actual}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.resultado_label.config(text=f"Error en ciclo {self.ciclo_actual}")

        # Preparar siguiente ciclo si continúa el análisis
        if self.is_analyzing and not self.is_closing:
            if GUARDAR_FOTOGRAMAS_ROI:
                try:
                    for file in os.listdir(self.roi_folder):
                        file_path = os.path.join(self.roi_folder, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            print(f"Error al eliminar {file_path}: {e}")
                    print(f"\n[LIMPIEZA] Buffer ROI limpiado para siguiente ciclo")
                except Exception as e:
                    print(f"Error al limpiar el buffer ROI: {e}")
            
            self.promedios_rgb = []
            self.tiempos = []
            self.tiempo_inicio = time.time()
            self.ciclo_actual += 1
            self.resultado_label.config(text=f"Iniciando ciclo {self.ciclo_actual}...")
            print(f"\n[NUEVO CICLO] Iniciando ciclo {self.ciclo_actual}...")
        elif self.is_closing:
            self.cerrar_aplicacion()
            
    def mostrar_resultados(self, resultados):
    """
    Muestra la frecuencia cardíaca calculada en la interfaz y en consola. 
    Recibe el diccionario de resultados, formatea el texto con el número de ciclo y el BPM, 
    actualiza la etiqueta de la interfaz y lo imprime en terminal.
    """
        if resultados:
            texto = f"Ciclo {self.ciclo_actual} - Frecuencia cardíaca: {resultados['verde']} BPM"
        else:
            texto = f"Ciclo {self.ciclo_actual} - No se pudo calcular la frecuencia cardíaca"
        
        # Mostrar en la interfaz gráfica
        self.resultado_label.config(text=texto)
        
        # Mostrar en la terminal
        print(texto)

        # Actualizar la interfaz gráfica inmediatamente
        self.window.update()

    def guardar_resultados(self, resultados):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"ciclo_{self.ciclo_actual}_{timestamp}_frecuencia.txt"
        
        with open(os.path.join(self.carpeta_usuario, nombre_archivo), "w") as f:
            f.write(f"Rojo: {resultados['rojo']} BPM\n")
            f.write(f"Verde: {resultados['verde']} BPM\n")
            f.write(f"Azul: {resultados['azul']} BPM\n")
            
            # Si tenemos datos de fase
            if 'fase' in resultados:
                f.write(f"\nAnálisis de Fase:\n")
                f.write(f"Frecuencia por fase: {resultados['fase']} BPM\n")
                f.write(f"Confianza: {resultados.get('confianza', 'N/A')}\n")
    
    def crear_grafica_fase(self, datos):
        """
        Crea una gráfica específica para visualizar los resultados del análisis de fase.
        Esta función debe llamarse desde crear_y_guardar_graficas cuando hay datos de fase.
        """
        if 'fase' not in datos:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear figura para visualización de fase
        fig = plt.figure(figsize=(12, 8))
        
        # Subplot 1: Señal filtrada y cruces por cero
        ax1 = plt.subplot(3, 1, 1)
        tiempo = np.arange(len(datos['promedio_filtrado_butter'])) / FPS
        ax1.plot(tiempo, datos['promedio_filtrado_butter'][:, 1], label='Señal filtrada', color='green')
        
        if 'cruces' in datos['fase']:
            cruces = datos['fase']['cruces']
            ax1.scatter(tiempo[cruces], datos['promedio_filtrado_butter'][cruces, 1], 
                    color='red', marker='o', label='Cruces por cero de fase')
        
        ax1.set_title('Señal filtrada con detección de ciclos por fase')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud')
        ax1.legend()
        ax1.grid(True)
        
        # Subplot 2: Fase instantánea
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(tiempo, datos['fase']['fase'], label='Fase instantánea', color='blue')
        ax2.set_title('Fase instantánea desenrollada')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Fase (rad)')
        ax2.grid(True)
        
        # Subplot 3: Frecuencia instantánea
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(tiempo, datos['fase']['frecuencia_instantanea'], label='Frecuencia (Hz)', color='purple')
        ax3.axhline(y=0.8, color='r', linestyle='--', label='Min FC (48 BPM)')
        ax3.axhline(y=3.0, color='r', linestyle='--', label='Max FC (180 BPM)')
        
        # Convertir Hz a BPM para el eje secundario
        ax3_bpm = ax3.twinx()
        min_freq, max_freq = ax3.get_ylim()
        ax3_bpm.set_ylim(min_freq * 60, max_freq * 60)
        ax3_bpm.set_ylabel('Frecuencia Cardíaca (BPM)')
        
        ax3.set_title('Frecuencia instantánea')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Frecuencia (Hz)')
        ax3.grid(True)
        ax3.legend(loc='upper left')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.carpeta_usuario, 
                            f"ciclo_{self.ciclo_actual}_{timestamp}_analisis_fase.png"))
        plt.close(fig)

    def crear_y_guardar_graficas(self, datos):
    """
    Guarda los resultados básicos de un ciclo en archivo de texto. 
    Recibe el diccionario con los BPM por canal (rojo, verde, azul). 
    Crea un archivo con timestamp en la carpeta del usuario, escribe los valores de cada canal 
    y incluye datos del análisis de fase si existen.
    """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(datos['tiempos'], datos['promedios_rgb'][:, 0], label='Rojo original', color='red')
        plt.plot(datos['tiempos'], datos['promedios_rgb'][:, 1], label='Verde original', color='green')
        plt.plot(datos['tiempos'], datos['promedios_rgb'][:, 2], label='Azul original', color='blue')
        plt.plot(datos['tiempos'], datos['promedio_filtrado'][:, 0], label='Rojo Kalman', linestyle='--', color='darkred')
        plt.plot(datos['tiempos'], datos['promedio_filtrado'][:, 1], label='Verde Kalman', linestyle='--', color='darkgreen')
        plt.plot(datos['tiempos'], datos['promedio_filtrado'][:, 2], label='Azul Kalman', linestyle='--', color='darkblue')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Valor promedio')
        plt.title(f'Ciclo {self.ciclo_actual} - Promedio RGB de la frente de la cara en cada fotograma (Kalman)')
        plt.legend()
        plt.grid(True)
        fig1.savefig(os.path.join(self.carpeta_usuario, f"ciclo_{self.ciclo_actual}_{timestamp}_grafica_kalman.png"))
        plt.close(fig1)

        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(datos['tiempos'], datos['promedio_filtrado_butter'][:, 0], label='Rojo Butterworth', linestyle='-', color='red')
        plt.plot(datos['tiempos'], datos['promedio_filtrado_butter'][:, 1], label='Verde Butterworth', linestyle='-', color='green')
        plt.plot(datos['tiempos'], datos['promedio_filtrado_butter'][:, 2], label='Azul Butterworth', linestyle='-', color='blue')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Valor promedio')
        plt.title(f'Ciclo {self.ciclo_actual} - Señal filtrada con Butterworth (0.8 Hz - 2.33 Hz)')
        plt.legend()
        plt.grid(True)
        fig2.savefig(os.path.join(self.carpeta_usuario, f"ciclo_{self.ciclo_actual}_{timestamp}_grafica_butterworth.png"))
        plt.close(fig2)

        fig3 = plt.figure(figsize=(10, 6))
        n = len(datos['xf'])
        plt.plot(datos['xf'][:n//2], np.abs(datos['yf_rojo'][:n//2]), label='Rojo FFT', color='red')
        plt.plot(datos['xf'][:n//2], np.abs(datos['yf_verde'][:n//2]), label='Verde FFT', color='green')
        plt.plot(datos['xf'][:n//2], np.abs(datos['yf_azul'][:n//2]), label='Azul FFT', color='blue')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.title(f'Ciclo {self.ciclo_actual} - FFT de la señal filtrada por Butterworth')
        plt.legend()
        plt.grid(True)
        fig3.savefig(os.path.join(self.carpeta_usuario, f"ciclo_{self.ciclo_actual}_{timestamp}_grafica_fft.png"))
        plt.close(fig3)

        # Añadir nueva gráfica de fase si está disponible
        if 'fase' in datos:
            self.crear_grafica_fase(datos)
            
    def mostrar_resultados_detallados(self, fusion_result):
        """Muestra los resultados detallados en la interfaz"""
        # Resultado principal
        bpm_final = fusion_result['bpm']
        std_final = fusion_result['std']
        conf_final = fusion_result['confidence']
        
        texto_principal = f"Ciclo {self.ciclo_actual} - FC: {bpm_final:.1f} ± {std_final:.1f} BPM (Confianza: {conf_final:.1%})"
        self.resultado_label.config(text=texto_principal)
        
        # Limpiar frame de detalles
        for widget in self.detail_frame.winfo_children():
            widget.destroy()
        
        # Agregar resultados por método
        for i, (method, result) in enumerate(fusion_result['methods_results'].items()):
            frame = ttk.Frame(self.detail_frame)
            frame.grid(row=i, column=0, sticky='ew', pady=2)
            
            # Nombre del método
            label_method = ttk.Label(frame, text=f"{method}:", width=12)
            label_method.grid(row=0, column=0, sticky='w')
            
            # BPM
            label_bpm = ttk.Label(frame, text=f"{result['bpm']:.1f} BPM")
            label_bpm.grid(row=0, column=1, padx=5)
            
            # Barra de confianza
            conf_frame = ttk.Frame(frame)
            conf_frame.grid(row=0, column=2, padx=5)
            
            conf_bar = ttk.Progressbar(conf_frame, length=100, mode='determinate')
            conf_bar['value'] = result['confidence'] * 100
            conf_bar.pack(side='left')
            
            conf_label = ttk.Label(conf_frame, text=f"{result['confidence']:.1%}")
            conf_label.pack(side='left', padx=5)
            
            # Peso en fusión
            if method in fusion_result['weights']:
                weight = fusion_result['weights'][method]
                weight_label = ttk.Label(frame, text=f"Peso: {weight:.2f}")
                weight_label.grid(row=0, column=3, padx=5)
        
        # Actualizar ventana
        self.window.update()

    def guardar_resultados_multimetodo(self, fusion_result):
        """Guarda los resultados de todos los métodos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"ciclo_{self.ciclo_actual}_{timestamp}_multimetodo.txt"
        
        with open(os.path.join(self.carpeta_usuario, nombre_archivo), "w") as f:
            f.write(f"=== CICLO {self.ciclo_actual} - RESULTADOS MULTI-MÉTODO ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resultado de fusión
            f.write("RESULTADO FINAL (FUSIÓN):\n")
            f.write(f"FC: {fusion_result['bpm']:.1f} ± {fusion_result['std']:.1f} BPM\n")
            f.write(f"Confianza global: {fusion_result['confidence']:.2%}\n\n")
            
            # Resultados individuales
            f.write("RESULTADOS POR MÉTODO:\n")
            for method, result in fusion_result['methods_results'].items():
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  BPM: {result['bpm']:.1f}\n")
                f.write(f"  Confianza: {result['confidence']:.2%}\n")
                if 'snr' in result:
                    f.write(f"  SNR: {result['snr']:.2f} dB\n")
                if 'alpha' in result:
                    f.write(f"  Alpha: {result['alpha']:.3f}\n")
                if method in fusion_result['weights']:
                    f.write(f"  Peso en fusión: {fusion_result['weights'][method]:.2%}\n")

    def crear_graficas_multimetodo(self, tiempos, promedios_rgb, promedio_filtrado, fusion_result):
        """Crea gráficas específicas para el análisis multi-método"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Gráfica 1: Señales originales y filtradas
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Señales originales
        ax1.plot(tiempos, promedios_rgb[:, 0], 'r-', alpha=0.5, label='Rojo')
        ax1.plot(tiempos, promedios_rgb[:, 1], 'g-', alpha=0.5, label='Verde')
        ax1.plot(tiempos, promedios_rgb[:, 2], 'b-', alpha=0.5, label='Azul')
        ax1.set_title(f'Ciclo {self.ciclo_actual} - Señales RGB Originales')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Intensidad')
        ax1.legend()
        ax1.grid(True)
        
        # Señales filtradas
        ax2.plot(tiempos, promedio_filtrado[:, 0], 'r-', label='Rojo (Kalman)')
        ax2.plot(tiempos, promedio_filtrado[:, 1], 'g-', label='Verde (Kalman)')
        ax2.plot(tiempos, promedio_filtrado[:, 2], 'b-', label='Azul (Kalman)')
        ax2.set_title('Señales Filtradas con Kalman')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Intensidad normalizada')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        fig1.savefig(os.path.join(self.carpeta_usuario, 
                                f"ciclo_{self.ciclo_actual}_{timestamp}_señales.png"))
        plt.close(fig1)
        
        # Gráfica 2: Comparación de señales de pulso por método
        fig2, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (method_name, result) in enumerate(fusion_result['methods_results'].items()):
            if idx < len(axes) and 'pulse_signal' in result:
                ax = axes[idx]
                signal = result['pulse_signal']
                
                # Señal en tiempo
                ax.plot(tiempos[:len(signal)], signal)
                ax.set_title(f'{method_name} - BPM: {result["bpm"]:.1f} (Conf: {result["confidence"]:.1%})')
                ax.set_xlabel('Tiempo (s)')
                ax.set_ylabel('Amplitud')
                ax.grid(True)
        
        # Ocultar ejes no usados
        for idx in range(len(fusion_result['methods_results']), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Ciclo {self.ciclo_actual} - Señales de Pulso por Método')
        plt.tight_layout()
        fig2.savefig(os.path.join(self.carpeta_usuario, 
                                f"ciclo_{self.ciclo_actual}_{timestamp}_pulsos.png"))
        plt.close(fig2)
        
        # Gráfica 3: Espectros de frecuencia
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        for idx, (method_name, result) in enumerate(fusion_result['methods_results'].items()):
            if 'pulse_signal' in result:
                signal = result['pulse_signal']
                # FFT
                n = len(signal)
                yf = fft(signal)
                xf = fftfreq(n, 1/FPS)
                
                # Solo frecuencias positivas
                mask = xf > 0
                ax.plot(xf[mask] * 60, np.abs(yf[mask]), 
                       color=colors[idx % len(colors)], 
                       label=f'{method_name} ({result["bpm"]:.0f} BPM)',
                       alpha=0.7)
        
        ax.set_xlim(40, 200)
        ax.set_xlabel('Frecuencia (BPM)')
        ax.set_ylabel('Amplitud')
        ax.set_title(f'Ciclo {self.ciclo_actual} - Espectros de Frecuencia por Método')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        fig3.savefig(os.path.join(self.carpeta_usuario, 
                                f"ciclo_{self.ciclo_actual}_{timestamp}_espectros.png"))
        plt.close(fig3)

    def generar_grafica_continua(self):
        """Genera gráfica de evolución temporal de todos los métodos"""
        if not self.ciclos:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfica superior: BPM por método
        # Primero los métodos originales para compatibilidad
        colores_originales = {
            'rojo': ('red', '--', 4),
            'verde': ('green', '--', 4), 
            'azul': ('blue', '--', 4)
        }
        
        # Luego los nuevos métodos
        colores_nuevos = {
            'fusion': ('black', '-', 8),
            'verde_fft': ('forestgreen', '-', 6), 
            'fase': ('navy', '-', 6),
            'pos': ('crimson', '-', 6),
            'chrom': ('purple', '-', 6),
            'ica': ('darkorange', '-', 6)
        }
        
        # Combinar todos los colores
        todos_colores = {**colores_originales, **colores_nuevos}
        
        for method, (color, linestyle, markersize) in todos_colores.items():
            if method in self.frecuencias_cardiacas and self.frecuencias_cardiacas[method]:
                valores = self.frecuencias_cardiacas[method]
                # Asegurar que tengamos el mismo número de valores que ciclos
                if len(valores) == len(self.ciclos):
                    label = method.upper() if method != 'verde_fft' else 'VERDE FFT'
                    ax1.plot(self.ciclos, valores, 'o-', 
                            color=color, 
                            linestyle=linestyle,
                            label=label, 
                            linewidth=2 if method == 'fusion' else 1,
                            markersize=markersize,
                            alpha=0.5 if method in colores_originales else 1.0)
        
        ax1.set_xlabel('Ciclo')
        ax1.set_ylabel('Frecuencia Cardíaca (BPM)')
        ax1.set_title('Evolución de FC - Comparación Multi-Método')
        ax1.legend(ncol=3, loc='upper right')
        ax1.grid(True)
        ax1.set_ylim(40, 120)
        ax1.set_xlim(0.5, max(self.ciclos) + 0.5)
        
        # Gráfica inferior: Confianza promedio
        if self.historial_resultados:
            confianzas = [r['confidence'] for r in self.historial_resultados]
            ax2.bar(self.ciclos, confianzas, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Ciclo')
            ax2.set_ylabel('Confianza Global')
            ax2.set_title('Confianza de las Mediciones')
            ax2.set_ylim(0, 1)
            ax2.grid(True, axis='y')
            
            # Agregar línea de tendencia de confianza
            if len(self.ciclos) > 1:
                z = np.polyfit(self.ciclos, confianzas, 1)
                p = np.poly1d(z)
                ax2.plot(self.ciclos, p(self.ciclos), "r--", alpha=0.8, label=f'Tendencia')
                ax2.legend()
        
        plt.tight_layout()
        filename = os.path.join(self.carpeta_usuario, "evolucion_fc_multimetodo.png")
        fig.savefig(filename)
        plt.close(fig)
        print(f"[GRÁFICA] Guardada evolución multi-método en: {filename}")

    def toggle_recuadro(self):
    #Bandera de visualización recuadro
        global dibujar_recuadro_verde
        dibujar_recuadro_verde = not dibujar_recuadro_verde

    def toggle_puntos(self):
    #Bandera de visualización puntos
        global dibujar_puntos
        dibujar_puntos = not dibujar_puntos

    def cerrar_aplicacion(self):
    """
    Finaliza la aplicación de forma ordenada. Activa la bandera de cierre, 
    detiene el análisis si está en curso, libera la cámara y destruye la ventana de Tkinter.
    """
        self.is_closing = True
        if self.is_analyzing:
            self.detener_analisis()
        if self.cap.isOpened():
            self.cap.release()
        self.window.quit()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "Análisis de frecuencia cardíaca")
    root.mainloop()
