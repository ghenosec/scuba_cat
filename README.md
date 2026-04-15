# Scuba Cat

Aplicação em Python que utiliza visão computacional para detectar um
gesto específico com as mãos e o rosto. Quando o gesto é reconhecido, um
overlay animado (GIF) é exibido na tela em tempo real.

------------------------------------------------------------------------

## Sobre o projeto

O sistema utiliza a webcam para capturar vídeo e aplica técnicas de
visão computacional para identificar:

-   Posição das mãos
-   Posição do nariz
-   Movimento da mão (gesto)

### Gesto necessário

Para ativar o reconhecimento:

1.  Uma mão deve estar tocando o nariz\
2.  A outra mão deve estar fechada (punho)\
3.  A mão fechada deve estar em movimento de "chacoalhar"

Quando o gesto é detectado corretamente, o scuba cat é exibido.

------------------------------------------------------------------------

## Estrutura do projeto

    .
    ├── scuba_cat/  
    │   ├── __init__.py
    │   ├── capture.py
    │   ├── config.py
    │   ├── face_detector.py
    │   ├── hand_tracker.py
    │   ├── overlay.py
    │   ├── recognizer.py
    │
    ├── assets/
    │   └── scuba_cat.gif
    │
    ├── main.py
    └── README.md

------------------------------------------------------------------------

## Requisitos

-   Python 3.11 ou 3.12\
-   Webcam funcional

> Observação: o MediaPipe não é compatível com Python 3.13 ou superior.

------------------------------------------------------------------------

## Instalação

### 1. Clone o repositório

``` bash
git clone https://github.com/ghenoset/scuba_cat.git
cd scuba-cat
```

### 2. Crie um ambiente virtual (recomendado)

``` bash
python -m venv venv
```

Ativação:

**Windows**

``` bash
venv\Scripts\activate
```

**Linux/Mac**

``` bash
source venv/bin/activate
```

------------------------------------------------------------------------

### 3. Instale as dependências

``` bash
pip install opencv-python mediapipe numpy pillow
```

------------------------------------------------------------------------

## Execução

``` bash
python main.py
```

------------------------------------------------------------------------

## Controles

  Tecla   Ação
  ------- ------------------------
  q       Encerrar aplicação
  d       Forçar animação
  r       Resetar reconhecimento

------------------------------------------------------------------------

## Configuração

Os principais parâmetros podem ser ajustados em:

    scuba_cat/config.py

Alguns exemplos:

-   `cooldown_s`: intervalo entre ativações
-   `overlay_duration_s`: duração do overlay
-   `min_detection_confidence`: sensibilidade da detecção

------------------------------------------------------------------------

## Customização

### Alterar o GIF

Substitua o arquivo:

    assets/scuba_cat.gif

------------------------------------------------------------------------

## Problemas comuns

### Webcam não inicia

-   Verifique se não está sendo usada por outro aplicativo
-   Teste outro índice de câmera em `config.py`

### MediaPipe não instala

-   Confirme a versão do Python (3.11 ou 3.12)

### Overlay não aparece

-   Verifique se o arquivo GIF está presente em `assets/`

