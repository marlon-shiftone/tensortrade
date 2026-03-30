# KuCoin Project Root

Este diretorio e a raiz operacional da integracao KuCoin.

Assim como em [`alpaca`](../alpaca), a logica reutilizavel do sistema fica no
pacote [`agent_trader`](../agent_trader), enquanto este diretorio guarda
estado, configuracoes e artefatos especificos da KuCoin.

Estrutura:

```text
kucoin/
  artifacts/
  configs/
  reports/
  state/
```

Observacoes importantes:

- O `ccxt` para `kucoin` spot nao expoe sandbox URL.
- Por isso, o modo `paper` aqui e um ledger local persistido em
  `state/paper_portfolio.json`.
- O modo `live` usa credenciais reais e continua exigindo aprovacao humana
  antes da ativacao.

Exemplos:

```bash
python -m agent_trader kucoin init-state
python -m agent_trader kucoin train --model-id kucoin_btc_v1 --symbol BTC/USDT
python -m agent_trader kucoin run --model-id kucoin_btc_v1 --mode paper
python -m agent_trader kucoin collect-metrics --model-id kucoin_btc_v1 --mode paper
python -m agent_trader kucoin operate --model-id kucoin_btc_v1 --mode paper --iterations 5
```
