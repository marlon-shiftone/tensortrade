# Dashboard

Dashboard operacional em Streamlit para visualizar os projetos operacionais do
`agent_trader`, como `alpaca/` e `kucoin/`.

## O que mostra

- estado atual do projeto
- status operacional destacado do loop online (`processed`, `waiting_new_bar`, `waiting_market_data`, `waiting_history`)
- registry de modelos
- snapshots de performance
- decisoes do supervisor
- execucoes recentes
- avaliacao shadow champion vs challenger
- historico de treinos

## Como rodar

Instale as dependencias do projeto e depois:

```bash
streamlit run dashboard/app.py
```

Ou usando a virtualenv local no Windows:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

## Observacoes

- a dashboard le direto os arquivos em `alpaca/`, `kucoin/` e qualquer outro
  diretorio operacional compativel no root do repositorio
- o refresh automatico e configuravel pela sidebar
- nao depende de banco de dados nem API adicional
