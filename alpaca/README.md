# Alpaca Project Root

Este diretorio e a raiz do projeto operacional do broker Alpaca.

## Motivo da separacao

O SDK oficial da Alpaca usa o pacote Python `alpaca`. Para evitar colisao de
imports, o codigo Python do sistema fica no pacote [`agent_trader`](../agent_trader),
enquanto este diretorio `alpaca/` guarda o estado e os artefatos operacionais
especificos da integracao Alpaca.

## Uso deste diretorio

- `state/`: estado persistido do registry e arquivos operacionais
- `artifacts/`: modelos, checkpoints e outros artefatos
- `reports/`: relatorios de paper/live e avaliacoes
- `configs/`: configuracoes operacionais por ambiente

## Pacote Python

O codigo executavel esta em [`agent_trader`](../agent_trader).

Exemplo:

```bash
python -m agent_trader alpaca init-state
```
