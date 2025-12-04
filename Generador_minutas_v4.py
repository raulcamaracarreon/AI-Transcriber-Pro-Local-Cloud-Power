import sys
import requests
import traceback
import json
import time
import textwrap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFileDialog, QSpinBox, 
    QRadioButton, QButtonGroup, QTextBrowser, QMessageBox, QGroupBox, 
    QScrollArea, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal

# --- 1. CONFIGURACIÓN ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
LOCAL_LLAMA_MODEL = "qwen2.5-coder:14b"  

PROMPT_PRESETS = {
    "🦜 Transcripción Literal (Fiel)": {
        "system": "Eres un editor de texto experto. Tu ÚNICA tarea es puntuar y corregir ortografía.",
        "user": "Formatea el siguiente fragmento de transcripción. \nREGLAS ESTRICTAS:\n1. NO resumas.\n2. NO omitas ninguna palabra.\n3. NO agregues notas como '[continúa igual]'.\n4. Solo agrega puntuación (puntos, comas, signos) y mayúsculas.\n5. Devuelve el texto COMPLETO.\n\nFragmento:"
    },
    "📝 Minuta Ejecutiva": {
        "system": "Eres un secretario ejecutivo experto y eficiente.",
        "user": "Analiza la siguiente transcripción y genera una MINUTA DE REUNIÓN. Debe incluir: \n1. Resumen breve.\n2. Puntos clave.\n3. Acuerdos tomados.\n4. Tareas pendientes.\n\nTranscripción:"
    },
    "🧹 Limpieza de Texto (Lectura)": {
        "system": "Eres un editor de texto profesional.",
        "user": "Reescribe para mejorar la legibilidad. Elimina muletillas (eh, um...), corrige gramática y separa en párrafos. NO resumas.\n\nTranscripción:"
    },
    "🔍 Extracción de Datos": {
        "system": "Eres un analista de datos preciso.",
        "user": "Extrae una lista con viñetas de: \n- Nombres propios.\n- Fechas y horas.\n- Lugares.\n- Herramientas mencionadas.\nNO hagas resumen, solo lista los datos.\n\nTranscripción:"
    },
    "✍️ Artículo de Blog": {
        "system": "Eres un redactor de contenido creativo (Copywriter).",
        "user": "Usa esta información para escribir un Artículo de Blog interesante con títulos y un tono conversacional.\n\nMaterial base:"
    }
}

# --- 2. WORKER THREAD (LÓGICA BLINDADA) ---
class TranscriptionWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    progress = Signal(int)

    def __init__(self, params):
        super().__init__()
        self.params = params 

    def run(self):
        try:
            transcription = ""
            p = self.params
            
            # --- FASE 1: TRANSCRIPCIÓN (WHISPER) ---
            self.progress.emit(5)
            self.status_update.emit("👂 Escuchando audio (Whisper Large-v3 Turbo)...")
            
            if p['provider'] == "openai":
                if not p['api_key']: raise ValueError("Falta la API Key de OpenAI.")
                with open(p['audio_path'], "rb") as audio_file:
                    resp = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {p['api_key']}"},
                        files={"file": ("audio.mp3", audio_file, "audio/mpeg")},
                        data={"model": "whisper-1"}
                    )
                if resp.status_code != 200: raise Exception(f"OpenAI Error: {resp.text}")
                transcription = resp.json().get("text", "")

            else: # Local Whisper
                if not p['local_pipeline']: raise ValueError("Modelo local no cargado.")
                
                # Ejecución optimizada: Forzamos español y usamos batching
                output = p['local_pipeline'](
                    p['audio_path'], 
                    batch_size=8,
                    generate_kwargs={"language": "spanish", "task": "transcribe"} 
                )
                transcription = output.get("text", "")

            if not transcription: raise Exception("Transcripción vacía.")
            
            self.progress.emit(40)
            self.status_update.emit("✅ Transcripción lista. Iniciando IA...")
            
            # --- FASE 2: PROCESAMIENTO (LLM CON CHUNKING) ---
            result_text = ""
            
            # Si se requiere fidelidad total, procesamos por partes
            if "Literal" in p['preset_name'] or "Limpieza" in p['preset_name']:
                chunks = textwrap.wrap(transcription, 3500, break_long_words=False, replace_whitespace=False)
                total_chunks = len(chunks)
                
                self.status_update.emit(f"🧠 Procesando {total_chunks} fragmentos de texto...")
                processed_chunks = []
                
                for i, chunk in enumerate(chunks):
                    current_prog = 40 + int((i / total_chunks) * 50)
                    self.progress.emit(current_prog)
                    self.status_update.emit(f"🧠 Analizando parte {i+1} de {total_chunks}...")
                    
                    chunk_response = self.call_llm(p, chunk)
                    processed_chunks.append(chunk_response)
                
                result_text = "\n\n".join(processed_chunks)
                
            else:
                # Modos de resumen
                self.progress.emit(60)
                self.status_update.emit("🧠 Analizando contenido global...")
                result_text = self.call_llm(p, transcription)

            # --- FASE 3: SALIDA ---
            self.progress.emit(95)
            self.status_update.emit("📝 Generando reporte final...")
            
            final_output = (
                f"# 🤖 RESULTADO ({p['preset_name']})\n\n{result_text}\n\n"
                f"{'='*60}\n"
                f"# 🎙️ TRANSCRIPCIÓN RAW (ORIGINAL)\n\n{transcription}"
            )
            
            self.progress.emit(100)
            self.finished.emit(final_output)

        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()

    def call_llm(self, p, text_input):
        if p['provider'] == "openai":
            messages = [
                {"role": "system", "content": p['system_prompt']},
                {"role": "user", "content": f"{p['user_prompt']}\n\n{text_input}"}
            ]
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {p['api_key']}", "Content-Type": "application/json"},
                json={"model": "gpt-4", "messages": messages, "max_tokens": p['max_tokens']}
            )
            if resp.status_code != 200: raise Exception(f"OpenAI Error: {resp.text}")
            return resp.json()['choices'][0]['message']['content']

        else: # Local Ollama
            payload = {
                "model": LOCAL_LLAMA_MODEL,
                "options": {"num_ctx": 8192},
                "messages": [
                    {"role": "system", "content": p['system_prompt']},
                    {"role": "user", "content": f"{p['user_prompt']}\n\n{text_input}"}
                ],
                "stream": False
            }
            resp = requests.post(OLLAMA_API_URL, json=payload)
            if resp.status_code != 200: raise Exception(f"Ollama Error: {resp.text}")
            return resp.json()['message']['content']

# --- 3. INTERFAZ GRÁFICA ---
class TranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Transcriber Pro - RTX Turbo Edition")
        self.setGeometry(100, 100, 1280, 850)
        self.apply_styles()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.left_container = QWidget()
        self.left_master_layout = QVBoxLayout(self.left_container)
        self.left_master_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_content = QWidget()
        self.left_scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.fixed_footer = QWidget()
        self.footer_layout = QVBoxLayout(self.fixed_footer)
        self.footer_layout.setContentsMargins(10, 0, 10, 10)

        self.left_master_layout.addWidget(self.scroll_area)
        self.left_master_layout.addWidget(self.fixed_footer)

        self.right_column = QWidget()
        self.right_layout = QVBoxLayout(self.right_column)

        main_layout.addWidget(self.left_container, 1)
        main_layout.addWidget(self.right_column, 2)

        self.setup_left_scrollable_content()
        self.setup_left_fixed_footer()
        self.setup_right_column()

        self.audio_path = ""
        self.local_whisper_pipeline = None
        self.worker = None

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f7fa; }
            QLabel { font-size: 13px; color: #2c3e50; font-weight: bold; margin-top: 5px; }
            QGroupBox { 
                background-color: white; border: 1px solid #dcdde1; border-radius: 6px; 
                margin-top: 10px; padding-top: 25px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 5px; color: #34495e; }
            QLineEdit, QComboBox, QSpinBox {
                border: 1px solid #bdc3c7; border-radius: 4px; padding: 8px; background-color: #fff; min-height: 20px;
            }
            QTextEdit { border: 1px solid #bdc3c7; border-radius: 4px; padding: 5px; background-color: #fff; }
            QProgressBar {
                border: 1px solid #bdc3c7; border-radius: 5px; text-align: center;
                background-color: #ecf0f1; height: 24px; color: #2c3e50; font-weight: bold;
            }
            QProgressBar::chunk { background-color: #3498db; width: 10px; margin: 0.5px; }
            QPushButton {
                background-color: #3498db; color: white; border-radius: 5px; padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton#GenerateBtn { background-color: #27ae60; font-size: 16px; padding: 15px; }
            QPushButton#GenerateBtn:hover { background-color: #219150; }
        """)

    def setup_left_scrollable_content(self):
        layout = self.left_scroll_layout
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)

        prov_group = QGroupBox("1. Configuración del Modelo")
        prov_layout = QVBoxLayout()
        self.openai_radio = QRadioButton("Nube (OpenAI API)")
        self.openai_radio.setChecked(True)
        self.local_radio = QRadioButton("Local (Ollama + Whisper)")
        prov_layout.addWidget(self.openai_radio)
        prov_layout.addWidget(self.local_radio)
        prov_layout.addWidget(QLabel("OpenAI API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-...")
        prov_layout.addWidget(self.api_key_input)
        
        prov_layout.addWidget(QLabel("Calidad Local (RTX 3060):"))
        self.quant_combo = QComboBox()
        # Nota para el usuario: "Default" ahora es FP16 (Rapidísimo)
        self.quant_combo.addItems(["Default (FP16 - Rápido)", "4-bit (Ahorra VRAM)", "8-bit"])
        prov_layout.addWidget(self.quant_combo)
        
        self.load_btn = QPushButton("Cargar Modelos")
        self.load_btn.clicked.connect(self.load_models)
        prov_layout.addWidget(self.load_btn)
        prov_group.setLayout(prov_layout)
        layout.addWidget(prov_group)

        file_group = QGroupBox("2. Archivo de Audio")
        file_layout = QVBoxLayout()
        self.file_btn = QPushButton("📂 Seleccionar Audio")
        self.file_btn.clicked.connect(self.select_audio)
        file_layout.addWidget(self.file_btn)
        self.file_label = QLabel("Ningún archivo seleccionado")
        self.file_label.setStyleSheet("font-weight: normal; color: #7f8c8d;")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        prompt_group = QGroupBox("3. Objetivo (Prompting)")
        prompt_layout = QVBoxLayout()
        prompt_layout.addWidget(QLabel("¿Qué quieres obtener?"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PROMPT_PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self.apply_preset) 
        prompt_layout.addWidget(self.preset_combo)
        prompt_layout.addWidget(QLabel("System Prompt:"))
        self.sys_edit = QTextEdit()
        self.sys_edit.setMaximumHeight(60)
        prompt_layout.addWidget(self.sys_edit)
        prompt_layout.addWidget(QLabel("User Prompt:"))
        self.user_edit = QTextEdit()
        self.user_edit.setMaximumHeight(80)
        prompt_layout.addWidget(self.user_edit)
        prompt_layout.addWidget(QLabel("Max Tokens:"))
        self.token_spin = QSpinBox()
        self.token_spin.setRange(500, 32000)
        self.token_spin.setValue(4000)
        prompt_layout.addWidget(self.token_spin)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        layout.addStretch()

    def setup_left_fixed_footer(self):
        self.status_lbl = QLabel("Listo para empezar")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-bottom: 5px;")
        self.footer_layout.addWidget(self.status_lbl)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% Completado")
        self.footer_layout.addWidget(self.progress_bar)

        self.gen_btn = QPushButton("🚀 PROCESAR AUDIO")
        self.gen_btn.setObjectName("GenerateBtn")
        self.gen_btn.clicked.connect(self.start_process)
        self.footer_layout.addWidget(self.gen_btn)
        
        self.apply_preset(self.preset_combo.currentText())

    def setup_right_column(self):
        self.out_browser = QTextBrowser()
        self.out_browser.setOpenExternalLinks(True)
        self.out_browser.setStyleSheet("background-color: white; border: 1px solid #dcdde1; padding: 15px; font-size: 14px;")
        
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("📝 Resultado:"))
        top_bar.addStretch()
        copy_btn = QPushButton("Copiar Todo")
        copy_btn.setFixedWidth(120)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.out_browser.toPlainText()))
        top_bar.addWidget(copy_btn)
        
        self.right_layout.addLayout(top_bar)
        self.right_layout.addWidget(self.out_browser)

    def apply_preset(self, preset_name):
        if preset_name in PROMPT_PRESETS:
            data = PROMPT_PRESETS[preset_name]
            self.sys_edit.setText(data["system"])
            self.user_edit.setText(data["user"])

    def select_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Audio", "", "Audio (*.mp3 *.wav *.m4a *.mp4)")
        if path:
            self.audio_path = path
            self.file_label.setText(f"...{path[-40:]}")

    def load_models(self):
        if self.local_radio.isChecked():
            self.status_lbl.setText("Cargando Whisper Local... (Esto tardará la primera vez)")
            self.progress_bar.setValue(0)
            self.load_btn.setEnabled(False)
            QApplication.processEvents()
            try:
                import torch
                from transformers import pipeline, BitsAndBytesConfig
                
                quant = self.quant_combo.currentText()
                model_kwargs = {}
                dtype = torch.float16 # Por defecto FP16 (Turbo para RTX 3060)

                if "4-bit" in quant:
                    model_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)}
                elif "8-bit" in quant:
                    model_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
                
                # --- MODELO "LARGE-V3-TURBO" ---
                # El mejor balance calidad/velocidad hoy en día.
                model_name = "openai/whisper-large-v3-turbo" 

                self.local_whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device_map="auto",
                    dtype=dtype, # Aceleración
                    model_kwargs=model_kwargs,
                    chunk_length_s=30, 
                    stride_length_s=5, 
                    return_timestamps=True
                )
                self.status_lbl.setText("✅ Whisper Cargado")
                self.progress_bar.setValue(100)
                QMessageBox.information(self, "Éxito", f"Modelo '{model_name}' cargado en GPU con precisión {dtype}.")
                
            except Exception as e:
                self.status_lbl.setText("❌ Error de Carga")
                self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
                QMessageBox.critical(self, "Error Fatal", str(e))
                print(e)
            finally:
                self.load_btn.setEnabled(True)
                self.progress_bar.setStyleSheet("")
        else:
            self.status_lbl.setText("✅ Modo Nube Listo")
            QMessageBox.information(self, "Info", "Modo Nube seleccionado.")

    def start_process(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Error", "Falta el archivo de audio.")
            return

        params = {
            "provider": "openai" if self.openai_radio.isChecked() else "local",
            "api_key": self.api_key_input.text().strip(),
            "audio_path": self.audio_path,
            "system_prompt": self.sys_edit.toPlainText(),
            "user_prompt": self.user_edit.toPlainText(),
            "max_tokens": self.token_spin.value(),
            "local_pipeline": self.local_whisper_pipeline,
            "preset_name": self.preset_combo.currentText()
        }

        self.gen_btn.setEnabled(False)
        self.out_browser.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("")
        
        self.worker = TranscriptionWorker(params)
        self.worker.status_update.connect(self.status_lbl.setText)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_finished(self, text):
        self.out_browser.setMarkdown(text)
        self.status_lbl.setText("✨ Finalizado")
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
        self.gen_btn.setEnabled(True)
        QMessageBox.information(self, "Proceso Terminado", "Tarea completada con éxito.")

    def on_error(self, err):
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
        QMessageBox.critical(self, "Error", err)
        self.gen_btn.setEnabled(True)
        self.status_lbl.setText("❌ Error")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    sys.exit(app.exec())