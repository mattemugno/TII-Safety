# Human Pose Estimation Examples

Un insieme di notebook e modelli per sperimentare diverse librerie di riconoscimento della posa umana.

## Struttura
├── data/ <br/>
│ └── *.jpg # Immagini di esempio <br/>
├── model/ <br/>
│ ├── pose_landmarker.task <br/>
│ └── intel/ <br/>
│ └── human-pose-estimation-0001/ <br/>
│ ├── .xml <br/>
│ └── .bin <br/>
├── yolo11n-pose.pt # Peso pre-allenato YOLOv11 <br/>
├── requirements.txt # Dipendenze Python <br/>
└── notebooks/ # Notebook indipendenti <br/>
├── Mediapipe.ipynb <br/> 
├── MoveNet.ipynb <br/>
├── OpenPose.ipynb <br/>
├── VitPose.ipynb <br/>
└── YOLOv11.ipynb <br/>

## Installazione

1. Crea e attiva un virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
2. Installa i pacchetti:
   ```bash
   pip install -r requirements.txt

## Esecuzione

Immagini di test: posiziona o sostituisci le immaigini in data/.

### Modelli:

I file _pose_landmarker.task_ (configurazione per il task di Landmarker di Mediapipe), 
_model/intel/human-pose-estimation-0001_ (modello di OpenVINO) e _yolo11n-pose.pt_ (pesi del modello YOLOv11) vengono 
scaricati automaticamente quando si esegue il codice nei notebook.

Notebook: ogni notebook permette di fare inferenza su una immagine usando un modello diverso:

* Mediapipe.ipynb
* MoveNet.ipynb
* OpenPose.ipynb
* VitPose.ipynb
* YOLOv11.ipynb

Ogni notebook carica automaticamente e indipendente immagini, modelli e librerie necessari.