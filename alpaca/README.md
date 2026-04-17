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

## Credenciais via .env

Voce pode colocar as credenciais em `alpaca/configs/.env`. O `agent_trader`
carrega automaticamente:

- `./.env`
- `alpaca/configs/.env`

Exemplo:

```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_DATA_FEED=iex
```

## Online Train

Para operar em paper, monitorar degradacao, retreinar challengers com janela
deslizante recente e promover automaticamente para paper quando o challenger
passar na avaliacao shadow:

```bash
python -m agent_trader alpaca online-train --model-id alpaca_aapl_v1 --iterations 0
```

Observacoes:

- `--iterations 0` significa loop continuo.
- o champion opera na paper account real da Alpaca
- o challenger e avaliado em `shadow paper` local
- a promocao para `paper_model_id` continua automatica apenas no ambiente paper
- `live` segue exigindo aprovacao humana
