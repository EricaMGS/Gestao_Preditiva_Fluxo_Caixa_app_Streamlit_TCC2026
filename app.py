import streamlit as st  # Interface do usuário
import pandas as pd  # Manipulação de dados
import numpy as np  # Cálculos
from sklearn.ensemble import RandomForestRegressor  # Modelo de IA
from sklearn.model_selection import train_test_split  # Divisão de teste
from sklearn.metrics import r2_score  # Métrica de precisão
import plotly.graph_objects as go  # Gráficos interativos

# Configurações de layout
st.set_page_config(page_title="Fluxo de Caixa IA", layout="wide")

# CSS para customizar a interface e remover textos em inglês
st.markdown("""
    <style>
    /* Remove o texto 'Drag and drop file here' */
    [data-testid="stFileUploadDropzone"] div div { display: none; }
    [data-testid="stFileUploadDropzone"]::before { content: "Arraste o arquivo Excel aqui ou clique"; font-size: 14px; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Gestão Preditiva de Fluxo de Caixa")

# 1. Carregar Arquivo
uploaded_file = st.sidebar.file_uploader("Carregar arquivo", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Lendo o Excel (neste modelo não pulamos linhas, o cabeçalho é na linha 0)
        df = pd.read_excel(uploaded_file)
        
        # 2. Limpeza de Dados (Data Science)
        # Convertendo a data (formato brasileiro DD/MM/YYYY)
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
        
        # Identificando a coluna de valor correta (Valor (R$))
        col_valor = 'Valor (R$)'
        
        # Removendo linhas onde o valor é vazio (como 'SALDO ANTERIOR')
        df = df.dropna(subset=[col_valor])
        
        # Separando Receitas (positivo) de Despesas (negativo)
        df['Receita'] = df[col_valor].apply(lambda x: x if x > 0 else 0)
        df['Despesa'] = df[col_valor].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Agrupando por dia para a IA aprender o comportamento diário
        df_diario = df.groupby('Data').agg({'Receita':'sum', 'Despesa':'sum'}).reset_index()
        
        # Criando Atributos Temporais (Features)
        df_diario['ano'] = df_diario['Data'].dt.year
        df_diario['mes'] = df_diario['Data'].dt.month
        df_diario['dia'] = df_diario['Data'].dt.day
        df_diario['dia_semana'] = df_diario['Data'].dt.dayofweek

        # 3. Inteligência Artificial (Machine Learning)
        X = df_diario[['ano', 'mes', 'dia', 'dia_semana']]
        
        # Treinando os modelos (Receita e Despesa)
        model_rec = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df_diario['Receita'])
        model_des = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df_diario['Despesa'])

        # 4. Testes de Eficácia (Score de Confiança)
        # O R² indica quanto o modelo entendeu o padrão dos seus dados
        score_rec = model_rec.score(X, df_diario['Receita'])
        score_des = model_des.score(X, df_diario['Despesa'])

        st.subheader("🎯 Testes de Eficácia")
        c1, c2, c3 = st.columns(3)
        c1.metric("Confiança Prev. Receita", f"{score_rec:.1%}")
        c2.metric("Confiança Prev. Despesa", f"{score_des:.1%}")
        c3.metric("Lançamentos Analisados", len(df))

        # 5. Previsão para os próximos 30 dias
        ultima_data = df_diario['Data'].max()
        datas_futuras = pd.date_range(ultima_data + pd.Timedelta(days=1), periods=30)
        df_futuro = pd.DataFrame({
            'ano': datas_futuras.year, 'mes': datas_futuras.month, 
            'dia': datas_futuras.day, 'dia_semana': datas_futuras.dayofweek
        })
        
        p_rec = model_rec.predict(df_futuro)
        p_des = model_des.predict(df_futuro)

        # 6. Gráfico de Fluxo de Caixa
        st.markdown("---")
        fig = go.Figure()
        
        # Histórico real
        fig.add_trace(go.Scatter(x=df_diario['Data'], y=df_diario['Receita'], name='Receita Atual', line=dict(color='#2ecc71')))
        fig.add_trace(go.Scatter(x=df_diario['Data'], y=df_diario['Despesa'], name='Despesa Atual', line=dict(color='#e74c3c')))
        
        # Previsão pontilhada
        fig.add_trace(go.Scatter(x=datas_futuras, y=p_rec, name='Estimativa Receita', line=dict(dash='dash', color='#27ae60')))
        fig.add_trace(go.Scatter(x=datas_futuras, y=p_des, name='Estimativa Despesa', line=dict(dash='dash', color='#c0392b')))

        fig.update_layout(title="Projeção para os Próximos 30 Dias", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Resumo final
        saldo_previsto = p_rec.sum() - p_des.sum()
        if saldo_previsto > 0:
            st.success(f"💰 Saldo Líquido Estimado para os próximos 30 dias: R$ {saldo_previsto:,.2f}")
        else:
            st.warning(f"⚠️ Atenção: Saldo Líquido Estimado Negativo: R$ {saldo_previsto:,.2f}")

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: Certifique-se que ele tem as colunas 'Data' e 'Valor (R$)'.")
        st.info(f"Detalhe do erro: {e}")

else:
    st.info("Aguardando o arquivo Excel para iniciar.")