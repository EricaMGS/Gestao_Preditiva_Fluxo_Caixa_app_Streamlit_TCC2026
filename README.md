# Gestao_Preditiva_Fluxo_Caixa_app_Streamlit_TCC2026
Aplicativo em Python com Streamlit para análise e previsão de fluxo de caixa, com utilização de técnicas de Ciência de Dados e aprendizado de máquina.
# Gestão Preditiva de Fluxo de Caixa com Python e Streamlit

Este repositório contém o desenvolvimento de um aplicativo para análise e previsão de fluxo de caixa, elaborado em Python com o framework Streamlit, como parte de um Trabalho de Conclusão de Curso (TCC).

A aplicação realiza a importação de dados financeiros em formato Excel, o tratamento e a organização das informações e a criação de variáveis temporais. Para a estimativa de receitas e despesas futuras, utiliza-se o algoritmo RandomForestRegressor, da biblioteca scikit-learn. Os resultados são apresentados por meio de visualizações gráficas interativas, oferecendo suporte à tomada de decisão financeira.

O projeto foi desenvolvido com boas práticas de engenharia de software, incluindo o uso de ambiente virtual (venv) e versionamento com Git. Os dados utilizados passaram por procedimentos de anonimização e ajuste, em conformidade com os princípios da Lei Geral de Proteção de Dados Pessoais (LGPD).

## Tecnologias Utilizadas
- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- Plotly  
- OpenPyXL  

## Execução da Aplicação

Após a instalação das dependências, a aplicação pode ser executada com o comando:

```bash
streamlit run app.py

