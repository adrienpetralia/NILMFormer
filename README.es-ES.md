

# NILMFormer

<div align="center">

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2502.15637&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2506.05880)
[![Python](https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12-blue)]()

</div>

Repositorio oficial de código PyTorch para **NILMFormer: Monitoreo de Carga No Intrusivo que Considera la No Estacionariedad** (ACM SIGKDD 2025).

📄 **Preimpresión en arXiv**: [NILMFormer](https://arxiv.org/abs/2506.05880).

<p align="center">
    <img width="300" src="https://github.com/adrienpetralia/NILMFormer/blob/main/assets/intro.png" alt="Intro">
</p>

Millones de medidores inteligentes han sido desplegados en todo el mundo, recolectando el consumo de energía de hogares individuales. Basándose en estas mediciones, las compañías eléctricas proporcionan retroalimentación sobre los comportamientos de consumo. Para ayudar a los clientes a comprender mejor su uso, los proveedores necesitan ofrecer retroalimentación **detallada** (por electrodoméstico), un problema desafiante conocido como **Monitoreo de Carga No Intrusivo (NILM)**.

El NILM tiene como objetivo desagregar el consumo total de energía de un hogar y recuperar el uso individual de diferentes electrodomésticos. Las soluciones actuales de última generación (SotA) se basan en aprendizaje profundo y procesan el consumo del hogar en subsecuencias. Sin embargo, los datos reales de medidores inteligentes son **no estacionarios**: los cambios de distribución dentro de cada segmento de ventana pueden afectar gravemente el rendimiento del modelo.

Presentamos **NILMFormer**, una arquitectura basada en Transformers de secuencia a secuencia diseñada para abordar este problema.

<p align="center">
    <img width="700" src="https://github.com/adrienpetralia/NILMFormer/blob/main/assets/results_sample.png" alt="Results sample">
</p>



## Contenido 📝

Este repositorio contiene el **código fuente** de NILMFormer, así como el código necesario para reproducir la evaluación experimental de nuestro artículo.  
También incluye **10 líneas base recientes de NILM de última generación (SotA)** reimplantadas en PyTorch.

---

### Primeros Pasos 🚀

Para instalar las dependencias, puede usar los siguientes comandos. ¡La vida es mucho más fácil gracias a [uv](https://astral.sh/blog/uv)!

```bash
pip install uv
git clone https://github.com/adrienpetralia/NILMFormer
cd NILMFormer
uv sync
```


### Estructura del Código 📁

```
.
├── assets                 # recursos para el archivo README 
├── configs                # carpeta de configuraciones (archivos '.yaml')
├── data                   # carpeta de información de datos
├── results                # carpeta de resultados detallados de experimentos
├── scripts                # scripts para ejecutar experimentos
│   ├── run_one_expe.py    #   script python para ejecutar un experimento
│   └── run_all_expe.sh    #   script bash para ejecutar todos los experimentos
├── src                    # paquete de código fuente
│   ├── helpers            #   funciones auxiliares (procesamiento, bucles de entrenamiento, métricas, ...)
│   ├── baselines          #   líneas base de NILM y series temporales
│   └── nilmformer         #   modelo nilmformer
├── pyproject.toml         # archivo de configuración del proyecto
└── uv.lock                # bloqueo para resolver dependencias
```

### Ejecutar un Experimento ⚙️

Para ejecutar un experimento **específico**, use el siguiente comando:
```
uv run -m scripts.run_one_expe \
    --dataset "UKDALE" \
    --sampling_rate "1min" \
    --appliance "WashingMachine" \
    --window_size 128 \
    --name_model NILMFormer \
    --seed 0
```

Para ejecutar **todos** los experimentos realizados en nuestro artículo (esto puede tardar un poco), use:
```
. scripts/run_all_expe.sh
```

## NILMFormer ⚡

**TL;DR** : **NILMFormer** es una **arquitectura basada en Transformers de secuencia a secuencia** diseñada específicamente para el **Monitoreo de Carga No Intrusivo (NILM)**. Aborda la naturaleza **no estacionaria** de los datos de medidores inteligentes dividiendo y codificando por separado la **forma**, la dinámica **temporal** y las **estadísticas intrínsecas** de cada subsecuencia. Estos componentes se fusionan luego dentro del bloque Transformer. Finalmente, la predicción se refina mediante una **transformación lineal** de las estadísticas de la serie de entrada, teniendo en cuenta la **pérdida de potencia** en el proceso de desagregación.

### Detalles de la Arquitectura 🔎

<p align="center">
    <img width="600" src="https://github.com/adrienpetralia/NILMFormer/blob/main/assets/nilmformer_details.png" alt="NILMFormer">
</p>

**Mecanismos para Manejar la No Estacionariedad:** Para abordar el aspecto no estacionario de los datos de consumo eléctrico, NILMFormer funciona primero estacionarizando la subsecuencia de entrada restando su media y desviación estándar.
Mientras que la subsecuencia normalizada se pasa a través de un bloque convolucional robusto que actúa como extractor de características, las estadísticas eliminadas se proyectan linealmente en un espacio de mayor dimensión (denominado *TokenStats*), y las marcas de tiempo son utilizadas por el módulo TimeRPE propuesto para calcular una matriz de codificación posicional.
Estas características se concatenan y se alimentan al bloque Transformer, seguido de un Head simple para obtener una secuencia unidimensional de valores.
El paso final consiste en proyectar linealmente de nuevo los *TokenStats* (denominados *ProjStats*) a 2 valores escalares que luego se utilizan para desnormalizar la salida, proporcionando el consumo final individual del electrodoméstico.

**Codificación Posicional Relacionada con Marcas de Tiempo (TimeRPE):** TimeRPE aprovecha las marcas de tiempo discretas (minutos, horas, días, meses) extraídas de cada subsecuencia de entrada. Cada marca de tiempo se transforma mediante una función sinusoidal, capturando comportamientos periódicos. 
Estas señales se proyectan luego en un espacio de mayor dimensión mediante una convolución 1D (tamaño del kernel = 1). 
Este enfoque proporciona una representación más **consciente del tiempo** que la codificación posicional estándar, ayudando al modelo a manejar mejor los patrones temporales del mundo real. 


## Contribuyentes 👨‍👨‍👦‍👦

* Adrien Petralia (Université Paris Cité, Investigación EDF)
* Philippe Charpentier (Investigación EDF)
* Youssef Kadhi (Investigación EDF)
* Themis Palpanas (IUF, Université Paris Cité) 

## Citar NILMFormer 📚

Si utiliza NILMFormer en su trabajo, por favor cite nuestro artículo:

```bibtex
@inproceedings{petralia2025nilmformer,
author = {Petralia, Adrien and Charpentier, Philippe and Kadhi, Youssef and Palpanas, Themis},
title = {NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity},
year = {2025},
publisher = {Association for Computing Machinery},
doi = {10.1145/3711896.3737251},
booktitle = {Proceedings of the 31th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining V.2},
location = {Toronto, ON, Canada},
series = {KDD '25}
}
```

## Agradecimientos 🎅

Trabajo apoyado por EDF R&D y el programa francés ANRT.
