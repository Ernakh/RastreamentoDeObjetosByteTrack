# Rastreamento de Objetos com Visão Computacional e Deep Learning

Este repositório apresenta uma implementação em Python para **rastreamento de múltiplos objetos em vídeo**, utilizando **Visão Computacional**, **Deep Learning**, **YOLO**, **OpenCV** e uma estratégia própria de associação inspirada no artigo **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**.

O projeto tem finalidade acadêmica e didática, buscando demonstrar como um sistema de rastreamento pode detectar objetos em cada frame de um vídeo e manter uma identidade persistente para cada objeto ao longo do tempo.

---

## Sobre o projeto

O rastreamento de objetos é uma das tarefas mais importantes da área de Visão Computacional. Diferente da simples detecção de objetos, em que o modelo identifica objetos isoladamente em uma imagem, o rastreamento busca acompanhar os mesmos objetos ao longo de uma sequência de frames.

Neste projeto, o processo é dividido em duas partes principais:

1. **Detecção de objetos**, realizada por um modelo YOLO.
2. **Rastreamento dos objetos**, realizado por uma implementação própria baseada em associação entre caixas delimitadoras.

A proposta é inspirada no funcionamento do ByteTrack, um método de rastreamento de múltiplos objetos que busca aproveitar não apenas detecções de alta confiança, mas também algumas detecções de baixa confiança que podem representar objetos reais em situações de oclusão, movimento rápido ou baixa visibilidade.

---

## Inspiração teórica

O artigo **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**, de Yifu Zhang et al., propõe uma abordagem simples e eficiente para rastreamento de múltiplos objetos.

A ideia central do artigo é que muitas abordagens tradicionais descartam automaticamente caixas de detecção com baixa confiança. No entanto, essas detecções fracas podem representar objetos reais, principalmente quando estão parcialmente ocultos, desfocados ou em movimento.

O ByteTrack propõe associar quase todas as caixas de detecção aos objetos já rastreados. Primeiro, são utilizadas as detecções de alta confiança. Depois, as detecções de baixa confiança são usadas para tentar recuperar objetos que não foram associados na primeira etapa.

Neste projeto, essa lógica foi adaptada de maneira simplificada e didática.

---

## Objetivo

O objetivo deste projeto é implementar um sistema de rastreamento de objetos que permita:

- detectar objetos em vídeos ou webcam;
- associar objetos entre frames consecutivos;
- manter IDs persistentes para os objetos rastreados;
- visualizar o rastreamento em tempo real;
- salvar o vídeo processado;
- servir como base para estudos acadêmicos sobre rastreamento de objetos, YOLO e ByteTrack.

---

## Tecnologias utilizadas

Este projeto utiliza as seguintes tecnologias e bibliotecas:

- Python
- OpenCV
- NumPy
- SciPy
- Ultralytics YOLO
- VS Code

---

## Como o sistema funciona

O funcionamento geral do sistema ocorre da seguinte forma:

1. O vídeo é carregado frame a frame utilizando OpenCV.
2. O modelo YOLO detecta os objetos presentes em cada frame.
3. Cada detecção gera uma caixa delimitadora, uma classe e um valor de confiança.
4. As detecções são separadas entre alta e baixa confiança.
5. O rastreador associa as detecções atuais aos objetos rastreados anteriormente.
6. A associação é feita utilizando IoU e o algoritmo Húngaro.
7. Cada objeto recebe um ID, que é mantido enquanto o objeto continuar sendo rastreado.
8. O resultado é exibido na tela e pode ser salvo em vídeo.

---

## Principais conceitos utilizados

### Detecção de objetos

A detecção de objetos consiste em identificar e localizar objetos em uma imagem ou frame de vídeo. Neste projeto, essa tarefa é realizada por um modelo YOLO, que retorna as caixas delimitadoras dos objetos detectados.

Cada caixa possui:

- coordenadas da região detectada;
- classe do objeto;
- confiança da detecção.

---

### Rastreamento de objetos

O rastreamento busca manter a identidade de um objeto ao longo dos frames. Por exemplo, se uma pessoa recebe o ID 1 em um frame, o sistema deve tentar manter esse mesmo ID nos frames seguintes.

Esse processo é importante em aplicações como:

- monitoramento por câmeras;
- contagem de pessoas;
- análise de tráfego;
- veículos autônomos;
- esportes;
- robótica;
- segurança;
- análise comportamental.

---

### IoU

O IoU, ou Intersection over Union, é uma métrica usada para medir a sobreposição entre duas caixas delimitadoras.

Quanto maior o IoU, maior é a semelhança espacial entre duas caixas. Neste projeto, o IoU é usado para decidir se uma detecção atual corresponde a um objeto rastreado anteriormente.

---

### Algoritmo Húngaro

O algoritmo Húngaro é utilizado para resolver o problema de associação entre objetos rastreados e novas detecções.

Ele busca encontrar a melhor combinação possível entre os objetos existentes e as caixas detectadas no frame atual.

---

### Detecções de alta e baixa confiança

Inspirado no ByteTrack, o sistema utiliza dois tipos de detecção:

- **Detecções de alta confiança:** usadas para associações mais seguras.
- **Detecções de baixa confiança:** usadas para tentar recuperar objetos que podem ter sido parcialmente ocultados ou detectados com menor certeza.

Essa estratégia ajuda a reduzir perdas de rastreamento em situações complexas.

---

## Instalação


Crie um ambiente virtual:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install ultralytics opencv-python scipy numpy
```


## Como executar

### Usando a webcam

Para executar o rastreamento utilizando a webcam:

```bash
python main.py --source 0 --classes 0
```

O parâmetro `--classes 0` indica que o sistema irá rastrear apenas pessoas.

---

### Usando um vídeo

Coloque um vídeo na pasta do projeto e execute:

```bash
python main.py --source video.mp4 --classes 0
```

---

### Salvando o resultado

Para salvar o vídeo processado, utilize o parâmetro `--save`:

```bash
python main.py --source video.mp4 --classes 0 --save
```

O vídeo será salvo com o nome:

```text
resultado_rastreamento.mp4
```

---

## Exemplos de uso

### Rastrear apenas pessoas

```bash
python main.py --source video.mp4 --classes 0
```

---

### Rastrear pessoas, carros e motos

```bash
python main.py --source video.mp4 --classes 0,2,3
```

---

### Usar a webcam

```bash
python main.py --source 0 --classes 0
```

---

### Salvar o vídeo processado

```bash
python main.py --source video.mp4 --classes 0 --save
```

---

### Usar outro modelo YOLO

```bash
python main.py --source video.mp4 --model yolov8n.pt --classes 0
```

Ou:

```bash
python main.py --source video.mp4 --model yolo11n.pt --classes 0
```

---

## Classes do conjunto COCO

Os modelos YOLO pré-treinados geralmente utilizam classes do conjunto COCO. Algumas classes comuns são:

| ID | Classe |
|---:|--------|
| 0 | pessoa |
| 1 | bicicleta |
| 2 | carro |
| 3 | moto |
| 5 | ônibus |
| 7 | caminhão |
| 16 | cachorro |
| 17 | cavalo |

Exemplo:

```bash
python main.py --source video.mp4 --classes 0,2,3
```

Esse comando rastreia pessoas, carros e motos.

---

## Principais partes do código

### Carregamento do modelo YOLO

```python
model = YOLO("yolo11n.pt")
```

O modelo YOLO é responsável por detectar os objetos em cada frame.

---

### Leitura do vídeo

```python
cap = cv2.VideoCapture(source)
```

O OpenCV é utilizado para abrir a webcam ou o arquivo de vídeo.

---

### Detecção dos objetos

```python
result = model(frame, verbose=False)[0]
```

A cada frame, o modelo realiza a detecção dos objetos presentes na imagem.

---

### Extração das caixas detectadas

```python
boxes = result.boxes.xyxy.cpu().numpy()
scores = result.boxes.conf.cpu().numpy()
classes = result.boxes.cls.cpu().numpy().astype(int)
```

Essas informações representam:

- `boxes`: coordenadas das caixas delimitadoras;
- `scores`: confiança das detecções;
- `classes`: classe de cada objeto detectado.

---

### Cálculo de IoU

```python
iou = bbox_iou(track.bbox, det["bbox"])
```

O IoU é usado para medir a semelhança entre a caixa de um objeto rastreado e uma nova detecção.

---

### Associação com algoritmo Húngaro

```python
row_indices, col_indices = linear_sum_assignment(cost_matrix)
```

O algoritmo Húngaro define a melhor associação entre os objetos já rastreados e as novas detecções.

---

### Criação de novos rastros

```python
new_track = Track(
    track_id=self.next_id,
    bbox=detection["bbox"],
    score=detection["score"],
    class_id=detection["class_id"],
    class_name=detection["class_name"],
)
```

Quando uma nova detecção não corresponde a nenhum objeto existente, um novo rastro é criado.

---

### Desenho dos resultados

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

O sistema desenha as caixas delimitadoras e os IDs dos objetos rastreados no vídeo.

---

## Parâmetros do rastreador

No código, o rastreador pode ser configurado com os seguintes parâmetros:

```python
tracker = SimpleByteTracker(
    high_conf=0.45,
    low_conf=0.10,
    high_iou_threshold=0.30,
    low_iou_threshold=0.20,
    max_age=30,
    min_hits=2,
)
```

### Descrição dos parâmetros

| Parâmetro | Descrição |
|----------|-----------|
| `high_conf` | Define o limiar para considerar uma detecção como alta confiança |
| `low_conf` | Define o limiar mínimo para considerar uma detecção fraca |
| `high_iou_threshold` | IoU mínimo para associar detecções de alta confiança |
| `low_iou_threshold` | IoU mínimo para associar detecções de baixa confiança |
| `max_age` | Número máximo de frames que um objeto pode ficar sem ser detectado antes de ser removido |
| `min_hits` | Número mínimo de detecções necessárias para confirmar um objeto |

---

## Possibilidades de experimentação

Este projeto pode ser utilizado como base para experimentos acadêmicos envolvendo:

- rastreamento de pessoas;
- rastreamento de veículos;
- análise de vídeos com oclusão;
- comparação entre diferentes modelos YOLO;
- análise de desempenho em FPS;
- comparação entre diferentes valores de confiança;
- comparação entre diferentes valores de IoU;
- contagem de objetos em vídeo;
- análise de perda de identidade;
- análise de troca de IDs.

---

## Sugestões de experimentos

### Experimento 1: variação do limiar de confiança

Alterar o valor de `high_conf` e observar o impacto no rastreamento:

```python
high_conf=0.25
high_conf=0.45
high_conf=0.65
```

Com esse experimento, é possível analisar se o sistema perde mais ou menos objetos ao variar o nível mínimo de confiança exigido.

---

### Experimento 2: variação do IoU

Alterar o valor de `high_iou_threshold`:

```python
high_iou_threshold=0.20
high_iou_threshold=0.30
high_iou_threshold=0.50
```

Esse experimento permite observar como a exigência de maior sobreposição entre caixas interfere na manutenção dos IDs.

---

### Experimento 3: comparação entre vídeos

Testar o sistema em vídeos diferentes, como:

- vídeo com poucas pessoas;
- vídeo com muitas pessoas;
- vídeo com objetos em movimento rápido;
- vídeo com oclusão;
- vídeo com câmera parada;
- vídeo com câmera em movimento.

---

### Experimento 4: comparação entre modelos YOLO

Testar diferentes modelos:

```bash
python main.py --source video.mp4 --model yolov8n.pt --classes 0
```

```bash
python main.py --source video.mp4 --model yolo11n.pt --classes 0
```

Modelos menores tendem a ser mais rápidos, enquanto modelos maiores podem apresentar melhor precisão.

---

### Possíveis objetivos específicos

- implementar a detecção de objetos com YOLO;
- implementar um rastreador baseado em associação por IoU;
- manter IDs persistentes para os objetos rastreados;
- testar o sistema em diferentes vídeos;
- analisar o impacto dos limiares de confiança e IoU;
- discutir as limitações da abordagem.

---

## Limitações

Esta implementação possui finalidade didática e não corresponde à implementação oficial completa do ByteTrack.

Algumas limitações são:

- não utiliza o Filtro de Kalman completo;
- não utiliza Re-ID para comparação de aparência dos objetos;
- pode perder IDs em casos de oclusão prolongada;
- pode trocar IDs quando objetos muito semelhantes se cruzam;
- depende da qualidade das detecções geradas pelo YOLO;
- pode apresentar dificuldades em vídeos com câmera muito instável;
- não realiza avaliação automática com métricas como MOTA, IDF1 e HOTA.

Mesmo com essas limitações, o projeto é útil para compreender os fundamentos do rastreamento de múltiplos objetos em vídeo.

---

## Diferença em relação ao ByteTrack oficial

O ByteTrack oficial utiliza uma implementação mais robusta, com detector treinado especificamente e técnicas adicionais para alcançar alto desempenho em bases como MOT17, MOT20, HiEve e BDD100K.

Este projeto é uma versão simplificada, criada para fins de estudo, com foco em clareza e compreensão dos principais conceitos.

A principal semelhança está na ideia de utilizar detecções de alta e baixa confiança no processo de associação.

---

## Resultados esperados

Ao executar o projeto, espera-se visualizar:

- caixas delimitadoras sobre os objetos detectados;
- IDs atribuídos aos objetos rastreados;
- manutenção dos IDs ao longo dos frames;
- rastreamento em tempo real ou próximo do tempo real;
- possibilidade de salvar o vídeo processado.

---

## Referência principal

ZHANG, Yifu et al. **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**. European Conference on Computer Vision, 2022.

---

## Licença

Este projeto é disponibilizado para fins acadêmicos e educacionais.
