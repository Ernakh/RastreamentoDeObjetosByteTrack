#Artigo de referência: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf
#instalar
# pip install ultralytics opencv-python scipy numpy

#para testear, rodar:
#rastrear apenas pessoas (classe 0):
#python main.py --source 0 --classes 0 

#python main.py --source video.mp4 --classes 0 --save
#rastrear pessoas, biciletas, carros e motos (classes 0,1, 2,3):
#python main.py --source video.mp4 --classes 1 --save

#full
#python main.py --source video2.mp4 --classes 0,1,2,3,5,7,16,17 --save


# Os modelos YOLO pré-treinados geralmente utilizam classes do conjunto COCO. Algumas classes comuns são:

# | ID | Classe |
# |---:|--------|
# | 0 | pessoa |
# | 1 | bicicleta |
# | 2 | carro |
# | 3 | moto |
# | 5 | ônibus |
# | 7 | caminhão |
# | 16 | cachorro |
# | 17 | cavalo |

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calcula o IoU entre duas caixas:
    box = [x1, y1, x2, y2]
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 1
    missed: int = 0

    def update(self, detection: Dict):
        self.bbox = detection["bbox"]
        self.score = detection["score"]
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]
        self.hits += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits and self.missed == 0


class SimpleByteTracker:
    """
    Versão didática inspirada no ByteTrack.

    Ideia:
    1. Separar detecções em alta e baixa confiança.
    2. Associar tracks existentes com detecções de alta confiança.
    3. Tentar recuperar tracks perdidas usando detecções de baixa confiança.
    4. Criar novos tracks com detecções fortes que sobraram.
    """

    def __init__(
        self,
        high_conf: float = 0.45,
        low_conf: float = 0.10,
        high_iou_threshold: float = 0.30,
        low_iou_threshold: float = 0.20,
        max_age: int = 30,
        min_hits: int = 2,
    ):
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.high_iou_threshold = high_iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: List[Track] = []
        self.next_id = 1

    def associate(
        self,
        tracks: List[Track],
        detections: List[Dict],
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associa tracks e detecções usando IoU + algoritmo Húngaro.

        Retorna:
        - matches: pares (índice_track, índice_detecção)
        - unmatched_tracks
        - unmatched_detections
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                if track.class_id != det["class_id"]:
                    iou = 0.0
                else:
                    iou = bbox_iou(track.bbox, det["bbox"])

                cost_matrix[t_idx, d_idx] = 1.0 - iou

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            iou = 1.0 - cost_matrix[row, col]

            if iou >= iou_threshold:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_detections.discard(col)

        return matches, list(unmatched_tracks), list(unmatched_detections)

    def update(self, detections: List[Dict]) -> List[Track]:
        high_detections = [
            det for det in detections
            if det["score"] >= self.high_conf
        ]

        low_detections = [
            det for det in detections
            if self.low_conf <= det["score"] < self.high_conf
        ]

        updated_track_indices = set()

        # 1. Associação principal com detecções de alta confiança
        matches_high, unmatched_track_indices, unmatched_high_indices = self.associate(
            self.tracks,
            high_detections,
            self.high_iou_threshold,
        )

        for track_idx, det_idx in matches_high:
            self.tracks[track_idx].update(high_detections[det_idx])
            updated_track_indices.add(track_idx)

        # 2. Associação secundária com detecções de baixa confiança
        remaining_tracks = [self.tracks[i] for i in unmatched_track_indices]

        matches_low, unmatched_remaining_indices, unmatched_low_indices = self.associate(
            remaining_tracks,
            low_detections,
            self.low_iou_threshold,
        )

        for relative_track_idx, det_idx in matches_low:
            real_track_idx = unmatched_track_indices[relative_track_idx]
            self.tracks[real_track_idx].update(low_detections[det_idx])
            updated_track_indices.add(real_track_idx)

        # 3. Tracks que não foram atualizadas são marcadas como perdidas
        for idx, track in enumerate(self.tracks):
            if idx not in updated_track_indices:
                track.mark_missed()

        # 4. Criar novos tracks apenas com detecções fortes não associadas
        used_high_detections = {det_idx for _, det_idx in matches_high}

        for det_idx, detection in enumerate(high_detections):
            if det_idx not in used_high_detections:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=detection["bbox"],
                    score=detection["score"],
                    class_id=detection["class_id"],
                    class_name=detection["class_name"],
                )
                self.tracks.append(new_track)
                self.next_id += 1

        # 5. Remover tracks perdidas por muito tempo
        self.tracks = [
            track for track in self.tracks
            if track.missed <= self.max_age
        ]

        visible_tracks = [
            track for track in self.tracks
            if track.is_confirmed(self.min_hits)
        ]

        return visible_tracks


def extract_detections(result, target_classes: Optional[List[int]] = None) -> List[Dict]:
    detections = []

    if result.boxes is None:
        return detections

    names = result.names

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for bbox, score, class_id in zip(boxes, scores, classes):
        if target_classes is not None and class_id not in target_classes:
            continue

        detections.append({
            "bbox": bbox.astype(np.float32),
            "score": float(score),
            "class_id": int(class_id),
            "class_name": names[int(class_id)],
        })

    return detections


def draw_tracks(frame, tracks: List[Track]):
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)

        color = (
            int((track.track_id * 37) % 255),
            int((track.track_id * 17) % 255),
            int((track.track_id * 29) % 255),
        )

        label = f"ID {track.track_id} | {track.class_name} | {track.score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def parse_classes(classes_text: Optional[str]) -> Optional[List[int]]:
    """
    Exemplo:
    --classes 0          rastreia apenas pessoas
    --classes 0,2,3      rastreia pessoas, carros e motos
    """
    if classes_text is None:
        return None

    return [int(c.strip()) for c in classes_text.split(",") if c.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Caminho do vídeo ou 0 para webcam."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Modelo YOLO. Exemplo: yolo11n.pt ou yolov8n.pt."
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="IDs das classes separadas por vírgula. Exemplo: 0 para pessoa."
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Salva o vídeo processado."
    )

    args = parser.parse_args()

    target_classes = parse_classes(args.classes)

    print("Carregando modelo YOLO...")
    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {args.source}")

    tracker = SimpleByteTracker(
        high_conf=0.45,
        low_conf=0.10,
        high_iou_threshold=0.30,
        low_iou_threshold=0.20,
        max_age=30,
        min_hits=2,
    )

    writer = None

    if args.save:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            "resultado_rastreamento.mp4",
            fourcc,
            fps,
            (width, height),
        )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame, verbose=False)[0]

        detections = extract_detections(
            result,
            target_classes=target_classes,
        )

        tracks = tracker.update(detections)

        draw_tracks(frame, tracks)

        cv2.imshow("Rastreamento de Objetos - YOLO + Tracker Proprio", frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    print("Finalizado.")

    if args.save:
        print("Vídeo salvo como resultado_rastreamento.mp4")


if __name__ == "__main__":
    main()