# MLRobotForex

Projeto experimental de pesquisa em Forex (EURUSD): agrega dados de candles de 1 minuto para múltiplos timeframes (M5/M15/M30/H1), calcula indicadores técnicos (SMA e RSI via TA-Lib) em várias escalas, gera sinais de ordem com take-profit e stop-loss derivados da estrutura de 1 hora e plota gráficos interativos com Plotly.

> ⚠️ Projeto de pesquisa estatística / backtest experimental. Não é um robô de execução em mercado real. Leia a seção [Bugs conhecidos](#bugs-conhecidos).

## Como funciona (`trading.py`)

| Etapa | Função | Descrição |
|---|---|---|
| 1. Carga | `le_csv_1min()` | Lê os candles OHLCV de 1 minuto de `EURUSD/EURUSD1.csv` e filtra apenas dias úteis |
| 2. Agregação | `cria_velas_para_tfs_superiores()` | Constrói velas M5, M15, M30 e H1 a partir do M1 |
| 3. Sinal | `calcula_tp_sl_ordem()` | Compara o preço atual com o fechamento H1 e gera a ordem (`1` = compra, `-1` = venda, `0` = neutro) com TP e SL baseados na estrutura H1 |
| 4. Indicadores | `cria_media_movel()` / `cria_rsi()` | SMAs em múltiplos períodos (curtos e 50/100/200) e RSI-14 em cada timeframe |
| 5. Visualização | `cria_grafico()` | Candlestick + marcadores de TP em Plotly |

O fluxo executa no final do arquivo: carga → agregação → sinais → indicadores → impressão do DataFrame resultante.

## Estrutura

```
MLRobotForex/
├── EURUSD/
│   ├── EURUSD1.csv     ← candles de 1 minuto (lidos pelo trading.py)
│   ├── EURUSD5.csv
│   ├── EURUSD15.csv
│   ├── EURUSD30.csv
│   ├── EURUSD60.csv    ← H1
│   ├── EURUSD240.csv   ← H4
│   └── EURUSD1440.csv  ← diário
├── trading.py
└── README.md
```

O script lê apenas o `EURUSD1.csv`; os timeframes superiores são agregados em memória a partir dele (os demais CSVs servem de base de referência).

## Como rodar

Requisitos: Python 3 e as libs `numpy`, `pandas`, `plotly` e `TA-Lib`.

```bash
pip install -r requirements.txt
python trading.py
```

> O pacote `TA-Lib` via pip exige a biblioteca C instalada (ex.: `make install` do [ta-lib](https://github.com/ta-lib/ta-lib-python#installation), ou `conda install -c conda-forge ta-lib`).

## Bugs conhecidos

Assumidos no próprio código-fonte (ver comentários finais em `trading.py`):

- Um único DataFrame com dados mistos de vários timeframes **distorce as SMAs e o RSI** — o ideal é separar um DataFrame por timeframe;
- `plotly.tools` foi removido no Plotly ≥ 5 (use `plotly.subplots.make_subplots`);
- `cria_grafico()` referencia colunas de OHLC que não existem no DataFrame carregado;
- O nome "MLRobot" sugere aprendizado de máquina, mas o repositório contém apenas a fase de pesquisa/preparação de features — nenhum modelo de ML é treinado aqui.

## Aviso legal

Uso exclusivo para estudo e pesquisa. Nada aqui é recomendação de investimento. Nenhum robô executa ordens automaticamente neste repositório.
