import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configurações de layout
st.set_page_config(
    page_title="Fluxo de Caixa IA - Previsão Preditiva",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #2ecc71;
    }
    .warning-metric {
        border-left-color: #e74c3c;
    }
    .info-metric {
        border-left-color: #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header"><h1 style="color: white;">📊 Gestão Preditiva de Fluxo de Caixa</h1><p style="color: white;">Previsão de Receitas e Despesas com Machine Learning</p></div>', unsafe_allow_html=True)

# Sidebar para configurações
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

# Opção de carregar modelos salvos ou treinar novos
opcao_modelo = st.sidebar.radio(
    "Escolha a fonte dos modelos:",
    ["📁 Carregar modelos salvos (.pkl)", "🔄 Treinar novos modelos com novo arquivo"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **Como usar:**
    1. Selecione a fonte dos modelos
    2. Carregue os arquivos necessários
    3. Visualize as previsões e indicadores
    4. Baixe os relatórios gerados
""")

# Funções auxiliares
def carregar_modelos(pasta_modelos):
    """Carrega os modelos salvos"""
    try:
        modelo_receitas = joblib.load(os.path.join(pasta_modelos, 'modelo_previsao_receitas.pkl'))
        modelo_despesas = joblib.load(os.path.join(pasta_modelos, 'modelo_previsao_despesas.pkl'))
        scaler_receitas = joblib.load(os.path.join(pasta_modelos, 'scaler_receitas.pkl'))
        scaler_despesas = joblib.load(os.path.join(pasta_modelos, 'scaler_despesas.pkl'))
        feature_cols_receitas = joblib.load(os.path.join(pasta_modelos, 'feature_cols_receitas.pkl'))
        feature_cols_despesas = joblib.load(os.path.join(pasta_modelos, 'feature_cols_despesas.pkl'))
        
        return {
            'modelo_receitas': modelo_receitas,
            'modelo_despesas': modelo_despesas,
            'scaler_receitas': scaler_receitas,
            'scaler_despesas': scaler_despesas,
            'feature_cols_receitas': feature_cols_receitas,
            'feature_cols_despesas': feature_cols_despesas
        }
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None

def criar_features_futuras(datas_futuras):
    """Cria features para datas futuras"""
    df_futuro = pd.DataFrame({'Data': datas_futuras})
    
    df_futuro['ano'] = df_futuro['Data'].dt.year.astype(int)
    df_futuro['mes'] = df_futuro['Data'].dt.month.astype(int)
    df_futuro['dia'] = df_futuro['Data'].dt.day.astype(int)
    df_futuro['dia_semana'] = df_futuro['Data'].dt.dayofweek.astype(int)
    df_futuro['trimestre'] = df_futuro['Data'].dt.quarter.astype(int)
    
    # Features cíclicas
    df_futuro['mes_sin'] = np.sin(2 * np.pi * df_futuro['mes'] / 12)
    df_futuro['mes_cos'] = np.cos(2 * np.pi * df_futuro['mes'] / 12)
    df_futuro['dia_semana_sin'] = np.sin(2 * np.pi * df_futuro['dia_semana'] / 7)
    df_futuro['dia_semana_cos'] = np.cos(2 * np.pi * df_futuro['dia_semana'] / 7)
    
    df_futuro['fim_semana'] = df_futuro['dia_semana'].isin([5, 6]).astype(int)
    df_futuro['dia_proporcao'] = df_futuro['dia'] / 31
    
    return df_futuro

def fazer_previsoes(modelos, datas_futuras, meses=12):
    """Faz previsões usando os modelos carregados"""
    df_futuro = criar_features_futuras(datas_futuras)
    
    # Previsão de receitas
    X_receitas = df_futuro[modelos['feature_cols_receitas']]
    X_receitas_scaled = modelos['scaler_receitas'].transform(X_receitas)
    previsao_receitas = modelos['modelo_receitas'].predict(X_receitas_scaled)
    previsao_receitas = np.maximum(previsao_receitas, 0)
    
    # Previsão de despesas
    X_despesas = df_futuro[modelos['feature_cols_despesas']]
    X_despesas_scaled = modelos['scaler_despesas'].transform(X_despesas)
    previsao_despesas = modelos['modelo_despesas'].predict(X_despesas_scaled)
    previsao_despesas = np.maximum(previsao_despesas, 0)
    
    # Criar DataFrame de previsões
    df_previsoes = pd.DataFrame({
        'Data': datas_futuras,
        'Receita_Prevista': previsao_receitas,
        'Despesa_Prevista': previsao_despesas,
        'Saldo_Diario': previsao_receitas - previsao_despesas
    })
    
    # Calcular saldo acumulado
    df_previsoes['Saldo_Acumulado'] = df_previsoes['Saldo_Diario'].cumsum()
    
    # Agrupar por mês
    df_previsoes['Ano'] = df_previsoes['Data'].dt.year
    df_previsoes['Mes'] = df_previsoes['Data'].dt.month
    df_previsoes['Mes_Nome'] = df_previsoes['Data'].dt.strftime('%b/%Y')
    
    resumo_mensal = df_previsoes.groupby(['Ano', 'Mes', 'Mes_Nome']).agg({
        'Receita_Prevista': 'sum',
        'Despesa_Prevista': 'sum',
        'Saldo_Diario': 'sum'
    }).reset_index()
    resumo_mensal.columns = ['Ano', 'Mes', 'Mês', 'Receitas', 'Despesas', 'Saldo']
    
    return df_previsoes, resumo_mensal

def treinar_novos_modelos(df_diario):
    """Treina novos modelos com os dados carregados"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    
    # Criar features
    df_features = df_diario.copy()
    df_features['ano'] = df_features['Data'].dt.year
    df_features['mes'] = df_features['Data'].dt.month
    df_features['dia'] = df_features['Data'].dt.day
    df_features['dia_semana'] = df_features['Data'].dt.dayofweek
    df_features['trimestre'] = df_features['Data'].dt.quarter
    df_features['mes_sin'] = np.sin(2 * np.pi * df_features['mes'] / 12)
    df_features['mes_cos'] = np.cos(2 * np.pi * df_features['mes'] / 12)
    df_features['dia_semana_sin'] = np.sin(2 * np.pi * df_features['dia_semana'] / 7)
    df_features['dia_semana_cos'] = np.cos(2 * np.pi * df_features['dia_semana'] / 7)
    df_features['fim_semana'] = df_features['dia_semana'].isin([5, 6]).astype(int)
    df_features['dia_proporcao'] = df_features['dia'] / 31
    
    feature_cols = ['ano', 'mes', 'dia', 'dia_semana', 'trimestre',
                    'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos',
                    'fim_semana', 'dia_proporcao']
    
    X = df_features[feature_cols]
    y_rec = df_features['Receita']
    y_des = df_features['Despesa']
    
    # Treinar modelos
    scaler_rec = RobustScaler()
    scaler_des = RobustScaler()
    X_rec_scaled = scaler_rec.fit_transform(X)
    X_des_scaled = scaler_des.fit_transform(X)
    
    modelo_receitas = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    modelo_despesas = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    modelo_receitas.fit(X_rec_scaled, y_rec)
    modelo_despesas.fit(X_des_scaled, y_des)
    
    # Calcular scores
    score_rec = modelo_receitas.score(X_rec_scaled, y_rec)
    score_des = modelo_despesas.score(X_des_scaled, y_des)
    
    return {
        'modelo_receitas': modelo_receitas,
        'modelo_despesas': modelo_despesas,
        'scaler_receitas': scaler_rec,
        'scaler_despesas': scaler_des,
        'feature_cols_receitas': feature_cols,
        'feature_cols_despesas': feature_cols,
        'score_rec': score_rec,
        'score_des': score_des
    }

def processar_arquivo_excel(uploaded_file):
    """Processa o arquivo Excel carregado"""
    df = pd.read_excel(uploaded_file)
    
    # Limpeza de dados
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    df['Valor'] = pd.to_numeric(df['Valor (R$)'], errors='coerce')
    
    # Remover linhas com valores nulos
    df = df.dropna(subset=['Data', 'Valor'])
    
    # Remover linhas de saldo
    df = df[~df['Lançamento'].str.contains('SALDO TOTAL DISPONÍVEL DIA', na=False)]
    
    # Separar receitas e despesas
    df['Receita'] = df['Valor'].apply(lambda x: x if x > 0 else 0)
    df['Despesa'] = df['Valor'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Agrupar por dia
    df_diario = df.groupby('Data').agg({
        'Receita': 'sum',
        'Despesa': 'sum',
        'Valor': lambda x: (x > 0).sum()  # número de transações
    }).reset_index()
    df_diario.columns = ['Data', 'Receita', 'Despesa', 'Num_Transacoes']
    
    # Preencher dias sem transações
    datas_completas = pd.date_range(start=df_diario['Data'].min(), 
                                    end=df_diario['Data'].max(), 
                                    freq='D')
    df_diario = df_diario.set_index('Data').reindex(datas_completas).reset_index()
    df_diario.rename(columns={'index': 'Data'}, inplace=True)
    df_diario[['Receita', 'Despesa', 'Num_Transacoes']] = df_diario[['Receita', 'Despesa', 'Num_Transacoes']].fillna(0)
    
    return df, df_diario

# ============================================
# Interface Principal
# ============================================

if opcao_modelo == "📁 Carregar modelos salvos (.pkl)":
    st.sidebar.subheader("📁 Carregar Modelos")
    
    # Opção de upload dos modelos
    upload_modelos = st.sidebar.checkbox("Fazer upload dos modelos (opcional)", value=False)
    
    if upload_modelos:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            modelo_rec_file = st.file_uploader("Modelo Receitas", type=['pkl'], key="rec")
        with col2:
            modelo_des_file = st.file_uploader("Modelo Despesas", type=['pkl'], key="des")
        
        scaler_rec_file = st.sidebar.file_uploader("Scaler Receitas", type=['pkl'], key="srec")
        scaler_des_file = st.sidebar.file_uploader("Scaler Despesas", type=['pkl'], key="sdes")
        
        if modelo_rec_file and modelo_des_file and scaler_rec_file and scaler_des_file:
            modelos = {
                'modelo_receitas': joblib.load(modelo_rec_file),
                'modelo_despesas': joblib.load(modelo_des_file),
                'scaler_receitas': joblib.load(scaler_rec_file),
                'scaler_despesas': joblib.load(scaler_des_file),
                'feature_cols_receitas': ['ano', 'mes', 'dia', 'dia_semana', 'trimestre',
                                          'mes_sin', 'mes_cos', 'dia_semana_sin', 
                                          'dia_semana_cos', 'fim_semana', 'dia_proporcao'],
                'feature_cols_despesas': ['ano', 'mes', 'dia', 'dia_semana', 'trimestre',
                                          'mes_sin', 'mes_cos', 'dia_semana_sin', 
                                          'dia_semana_cos', 'fim_semana', 'dia_proporcao']
            }
            st.sidebar.success("✅ Modelos carregados com sucesso!")
        else:
            st.sidebar.warning("⚠️ Faça upload de todos os arquivos .pkl")
            modelos = None
    else:
        # Tentar carregar da pasta padrão
        pasta_padrao = r"C:\Users\Acer\Desktop\fluxo_caixa_os\Gestao_Preditiva_Fluxo_Caixa_app_Streamlit_TCC2026"
        if os.path.exists(pasta_padrao):
            modelos = carregar_modelos(pasta_padrao)
            if modelos:
                st.sidebar.success(f"✅ Modelos carregados de: {pasta_padrao}")
        else:
            st.sidebar.warning("⚠️ Nenhum modelo encontrado na pasta padrão")
            modelos = None
    
    # Se tiver modelos, carregar dados para previsão
    if modelos:
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader("Carregar arquivo Excel com histórico", type=['xlsx'])
        
        if uploaded_file:
            # Processar arquivo
            df_raw, df_diario = processar_arquivo_excel(uploaded_file)
            
            # Definir período de previsão
            meses_previsao = st.sidebar.slider("Período de previsão (meses)", 3, 24, 12)
            
            # Fazer previsões
            ultima_data = df_diario['Data'].max()
            datas_futuras = pd.date_range(start=ultima_data + timedelta(days=1), 
                                         periods=meses_previsao * 30, 
                                         freq='D')
            
            df_previsoes, resumo_mensal = fazer_previsoes(modelos, datas_futuras, meses_previsao)
            
            # ============================================
            # Exibir indicadores
            # ============================================
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_receitas_hist = df_diario['Receita'].sum()
                st.metric("💰 Receitas Históricas", f"R$ {total_receitas_hist:,.2f}")
            
            with col2:
                total_despesas_hist = df_diario['Despesa'].sum()
                st.metric("📉 Despesas Históricas", f"R$ {total_despesas_hist:,.2f}")
            
            with col3:
                saldo_hist = total_receitas_hist - total_despesas_hist
                st.metric("📊 Saldo Histórico", f"R$ {saldo_hist:,.2f}", 
                         delta=f"{saldo_hist/total_receitas_hist*100:.1f}%" if total_receitas_hist > 0 else None)
            
            with col4:
                total_receitas_prev = resumo_mensal['Receitas'].sum()
                total_despesas_prev = resumo_mensal['Despesas'].sum()
                saldo_prev = total_receitas_prev - total_despesas_prev
                st.metric("🔮 Saldo Previsto", f"R$ {saldo_prev:,.2f}", 
                         delta=f"{saldo_prev/total_receitas_prev*100:.1f}%" if total_receitas_prev > 0 else None,
                         delta_color="normal")
            
            st.markdown("---")
            
            # Métricas detalhadas
            st.subheader("📈 Indicadores de Desempenho")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                media_receita_diaria = df_diario['Receita'].mean()
                st.metric("Média Receita Diária", f"R$ {media_receita_diaria:.2f}")
            
            with col2:
                media_despesa_diaria = df_diario['Despesa'].mean()
                st.metric("Média Despesa Diária", f"R$ {media_despesa_diaria:.2f}")
            
            with col3:
                dias_positivos = len(df_diario[df_diario['Receita'] > 0])
                st.metric("Dias com Receita", f"{dias_positivos} dias", 
                         delta=f"{dias_positivos/len(df_diario)*100:.1f}%")
            
            with col4:
                melhor_dia = df_diario.loc[df_diario['Receita'].idxmax(), 'Data'].strftime('%d/%m/%Y')
                st.metric("Melhor Dia de Receita", melhor_dia,
                         delta=f"R$ {df_diario['Receita'].max():,.2f}")
            
            st.markdown("---")
            
            # Gráfico 1: Série temporal histórica + previsão
            st.subheader("📊 Projeção de Fluxo de Caixa")
            
            fig1 = go.Figure()
            
            # Histórico (últimos 90 dias para não poluir)
            historico_plot = df_diario.tail(90)
            fig1.add_trace(go.Scatter(
                x=historico_plot['Data'], 
                y=historico_plot['Receita'], 
                name='Receita Histórica',
                line=dict(color='#2ecc71', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.1)'
            ))
            fig1.add_trace(go.Scatter(
                x=historico_plot['Data'], 
                y=historico_plot['Despesa'], 
                name='Despesa Histórica',
                line=dict(color='#e74c3c', width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)'
            ))
            
            # Previsão
            fig1.add_trace(go.Scatter(
                x=df_previsoes['Data'], 
                y=df_previsoes['Receita_Prevista'], 
                name='Receita Prevista',
                line=dict(color='#27ae60', width=2, dash='dash')
            ))
            fig1.add_trace(go.Scatter(
                x=df_previsoes['Data'], 
                y=df_previsoes['Despesa_Prevista'], 
                name='Despesa Prevista',
                line=dict(color='#c0392b', width=2, dash='dash')
            ))
            
            fig1.update_layout(
                title="Evolução Histórica e Previsão",
                xaxis_title="Data",
                yaxis_title="Valor (R$)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Gráfico 2: Saldo acumulado
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=df_previsoes['Data'],
                y=df_previsoes['Saldo_Acumulado'],
                name='Saldo Acumulado Previsto',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig2.update_layout(
                title="📈 Projeção de Saldo Acumulado",
                xaxis_title="Data",
                yaxis_title="Saldo Acumulado (R$)",
                template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Gráfico 3: Comparativo mensal
            st.subheader("📅 Comparativo Mensal Previsto")
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Bar(
                x=resumo_mensal['Mês'],
                y=resumo_mensal['Receitas'],
                name='Receitas',
                marker_color='#2ecc71'
            ))
            fig3.add_trace(go.Bar(
                x=resumo_mensal['Mês'],
                y=resumo_mensal['Despesas'],
                name='Despesas',
                marker_color='#e74c3c'
            ))
            
            fig3.update_layout(
                barmode='group',
                title="Previsão Mensal",
                xaxis_title="Mês",
                yaxis_title="Valor (R$)",
                template="plotly_white"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Gráfico 4: Saldo mensal
            fig4 = go.Figure()
            
            colors = ['#2ecc71' if s >= 0 else '#e74c3c' for s in resumo_mensal['Saldo']]
            fig4.add_trace(go.Bar(
                x=resumo_mensal['Mês'],
                y=resumo_mensal['Saldo'],
                name='Saldo Mensal',
                marker_color=colors
            ))
            
            fig4.update_layout(
                title="💰 Saldo Mensal Previsto",
                xaxis_title="Mês",
                yaxis_title="Saldo (R$)",
                template="plotly_white"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Tabela de previsões
            st.subheader("📋 Detalhamento das Previsões Mensais")
            
            # Formatar tabela
            tabela_display = resumo_mensal.copy()
            tabela_display['Receitas'] = tabela_display['Receitas'].apply(lambda x: f"R$ {x:,.2f}")
            tabela_display['Despesas'] = tabela_display['Despesas'].apply(lambda x: f"R$ {x:,.2f}")
            tabela_display['Saldo'] = tabela_display['Saldo'].apply(lambda x: f"R$ {x:,.2f}")
            tabela_display = tabela_display.drop(['Ano', 'Mes'], axis=1)
            
            st.dataframe(tabela_display, use_container_width=True)
            
            # Botão para download dos resultados
            st.subheader("📥 Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download da tabela mensal
                csv_mensal = resumo_mensal.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Baixar Previsões Mensais (CSV)",
                    data=csv_mensal,
                    file_name=f"previsoes_mensais_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download das previsões diárias
                csv_diario = df_previsoes[['Data', 'Receita_Prevista', 'Despesa_Prevista', 'Saldo_Diario', 'Saldo_Acumulado']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📅 Baixar Previsões Diárias (CSV)",
                    data=csv_diario,
                    file_name=f"previsoes_diarias_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Resumo final
            st.markdown("---")
            st.subheader("📌 Resumo Executivo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card success-metric">
                    <strong>✅ Receitas Previstas</strong><br>
                    <span style="font-size: 24px;">R$ {total_receitas_prev:,.2f}</span><br>
                    <small>Total para {meses_previsao} meses</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card warning-metric">
                    <strong>⚠️ Despesas Previstas</strong><br>
                    <span style="font-size: 24px;">R$ {total_despesas_prev:,.2f}</span><br>
                    <small>Total para {meses_previsao} meses</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                cor = "success-metric" if saldo_prev > 0 else "warning-metric"
                st.markdown(f"""
                <div class="metric-card {cor}">
                    <strong>💵 Saldo Previsto</strong><br>
                    <span style="font-size: 24px;">R$ {saldo_prev:,.2f}</span><br>
                    <small>Para os próximos {meses_previsao} meses</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Alertas
            if saldo_prev < 0:
                st.warning(f"⚠️ **Atenção!** O saldo previsto para os próximos {meses_previsao} meses é negativo (R$ {saldo_prev:,.2f}). Recomenda-se rever as despesas ou buscar aumentar as receitas.")
            elif saldo_prev < total_receitas_prev * 0.1:
                st.info(f"ℹ️ O saldo previsto representa apenas {saldo_prev/total_receitas_prev*100:.1f}% das receitas totais. Considere criar uma reserva de emergência.")
            else:
                st.success(f"✅ Excelente! O saldo previsto é positivo e representa {saldo_prev/total_receitas_prev*100:.1f}% das receitas.")
                
        else:
            st.info("📂 Carregue um arquivo Excel com o histórico de transações para visualizar as previsões.")
    else:
        st.error("❌ Nenhum modelo foi carregado. Verifique se os arquivos .pkl estão na pasta correta ou faça o upload manualmente.")

else:  # Treinar novos modelos
    st.sidebar.subheader("📂 Carregar Arquivo")
    uploaded_file = st.sidebar.file_uploader("Carregar arquivo Excel", type=['xlsx'])
    
    if uploaded_file:
        # Processar arquivo
        df_raw, df_diario = processar_arquivo_excel(uploaded_file)
        
        # Treinar modelos
        with st.spinner("Treinando modelos de Machine Learning..."):
            modelos = treinar_novos_modelos(df_diario)
        
        # Exibir métricas de confiança
        st.subheader("🎯 Testes de Eficácia")
        col1, col2, col3 = st.columns(3)
        col1.metric("Confiança Previsão Receitas", f"{modelos['score_rec']:.1%}")
        col2.metric("Confiança Previsão Despesas", f"{modelos['score_des']:.1%}")
        col3.metric("Lançamentos Analisados", len(df_raw))
        
        st.info("""
        **Interpretação das métricas:**
        - Quanto maior a confiança (próximo a 100%), mais o modelo conseguiu aprender os padrões dos seus dados
        - Valores acima de 70% indicam boa capacidade de previsão
        - Valores abaixo de 50% sugerem que os dados podem ter pouca regularidade ou muitos outliers
        """)
        
        # Definir período de previsão
        meses_previsao = st.slider("Período de previsão (meses)", 3, 24, 12)
        
        # Fazer previsões
        ultima_data = df_diario['Data'].max()
        datas_futuras = pd.date_range(start=ultima_data + timedelta(days=1), 
                                     periods=meses_previsao * 30, 
                                     freq='D')
        
        df_previsoes, resumo_mensal = fazer_previsoes(modelos, datas_futuras, meses_previsao)
        
        # Salvar modelos (opcional)
        st.subheader("💾 Salvar Modelos Treinados")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Salvar Modelos como .pkl"):
                pasta_salvar = os.path.join(os.path.expanduser("~"), "Desktop", "modelos_fluxo_caixa")
                os.makedirs(pasta_salvar, exist_ok=True)
                
                joblib.dump(modelos['modelo_receitas'], os.path.join(pasta_salvar, 'modelo_previsao_receitas.pkl'))
                joblib.dump(modelos['modelo_despesas'], os.path.join(pasta_salvar, 'modelo_previsao_despesas.pkl'))
                joblib.dump(modelos['scaler_receitas'], os.path.join(pasta_salvar, 'scaler_receitas.pkl'))
                joblib.dump(modelos['scaler_despesas'], os.path.join(pasta_salvar, 'scaler_despesas.pkl'))
                
                st.success(f"✅ Modelos salvos em: {pasta_salvar}")
        
        # Exibir gráficos e resultados (similar à parte anterior)
        # ... (copiar a seção de exibição de gráficos da parte anterior)
        
        # Gráficos e resultados (mesmo código da parte anterior)
        st.subheader("📊 Projeção de Fluxo de Caixa")
        
        fig1 = go.Figure()
        historico_plot = df_diario.tail(90)
        fig1.add_trace(go.Scatter(x=historico_plot['Data'], y=historico_plot['Receita'], 
                                  name='Receita Histórica', line=dict(color='#2ecc71', width=2)))
        fig1.add_trace(go.Scatter(x=historico_plot['Data'], y=historico_plot['Despesa'], 
                                  name='Despesa Histórica', line=dict(color='#e74c3c', width=2)))
        fig1.add_trace(go.Scatter(x=df_previsoes['Data'], y=df_previsoes['Receita_Prevista'], 
                                  name='Receita Prevista', line=dict(color='#27ae60', width=2, dash='dash')))
        fig1.add_trace(go.Scatter(x=df_previsoes['Data'], y=df_previsoes['Despesa_Prevista'], 
                                  name='Despesa Prevista', line=dict(color='#c0392b', width=2, dash='dash')))
        fig1.update_layout(title="Evolução Histórica e Previsão", xaxis_title="Data", 
                          yaxis_title="Valor (R$)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Saldo acumulado
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_previsoes['Data'], y=df_previsoes['Saldo_Acumulado'],
                                  name='Saldo Acumulado Previsto', line=dict(color='#3498db', width=2),
                                  fill='tozeroy'))
        fig2.update_layout(title="📈 Projeção de Saldo Acumulado", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Tabela de previsões
        st.subheader("📋 Previsões Mensais")
        tabela_display = resumo_mensal.copy()
        tabela_display['Receitas'] = tabela_display['Receitas'].apply(lambda x: f"R$ {x:,.2f}")
        tabela_display['Despesas'] = tabela_display['Despesas'].apply(lambda x: f"R$ {x:,.2f}")
        tabela_display['Saldo'] = tabela_display['Saldo'].apply(lambda x: f"R$ {x:,.2f}")
        st.dataframe(tabela_display.drop(['Ano', 'Mes'], axis=1), use_container_width=True)
        
        # Resumo final
        total_receitas = resumo_mensal['Receitas'].sum()
        total_despesas = resumo_mensal['Despesas'].sum()
        saldo_total = total_receitas - total_despesas
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Receitas Totais Previstas", f"R$ {total_receitas:,.2f}")
        col2.metric("Despesas Totais Previstas", f"R$ {total_despesas:,.2f}")
        col3.metric("Saldo Total Previsto", f"R$ {saldo_total:,.2f}", 
                   delta=f"{saldo_total/total_receitas*100:.1f}%" if total_receitas > 0 else None)
        
        if saldo_total < 0:
            st.warning(f"⚠️ **Atenção!** O saldo previsto para os próximos {meses_previsao} meses é negativo.")
    else:
        st.info("📂 Carregue um arquivo Excel para treinar os modelos e gerar previsões.")