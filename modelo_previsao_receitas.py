# ============================================
# MODELO DE PREVISÃO DE RECEITAS E DESPESAS - VERSÃO CORRIGIDA
# Correção: tratamento de tipos de dados e conversão para inteiros
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Modelos de Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Pré-processamento e métricas
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Para salvar os modelos
import joblib
import os

# Configurar visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("📊 MODELO DE PREVISÃO DE RECEITAS E DESPESAS - VERSÃO CORRIGIDA")
print("=" * 60)


def carregar_dados_completos(caminho_arquivo):
    """
    Carrega e prepara os dados separando receitas e despesas
    """
    print("\n📂 Carregando dados...")
    
    df = pd.read_excel(caminho_arquivo)
    print(f"   ✅ Arquivo carregado: {caminho_arquivo}")
    print(f"   📊 Total de registros: {len(df)}")
    
    # Converter data
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    
    # Converter valores
    df['Valor'] = pd.to_numeric(df['Valor (R$)'], errors='coerce')
    
    # Remover linhas com SALDO TOTAL DISPONÍVEL DIA (não são transações reais)
    df = df[~df['Lançamento'].str.contains('SALDO TOTAL DISPONÍVEL DIA', na=False)].copy()
    
    # Remover valores nulos
    df = df.dropna(subset=['Data', 'Valor'])
    
    # Separar receitas (valores positivos) e despesas (valores negativos)
    df_receitas = df[df['Valor'] > 0].copy()
    df_despesas = df[df['Valor'] < 0].copy()
    
    # Para despesas, converter para valores positivos (facilita análise)
    df_despesas['Valor'] = df_despesas['Valor'].abs()
    
    print(f"\n   💰 Receitas: {len(df_receitas)} registros, Total: R$ {df_receitas['Valor'].sum():,.2f}")
    print(f"   📉 Despesas: {len(df_despesas)} registros, Total: R$ {df_despesas['Valor'].sum():,.2f}")
    print(f"   📈 Saldo: R$ {(df_receitas['Valor'].sum() - df_despesas['Valor'].sum()):,.2f}")
    
    return df_receitas, df_despesas


def criar_features_diarias(df, nome):
    """
    Cria features temporais para modelagem diária
    """
    print(f"\n🔄 Criando features diárias para {nome}...")
    
    if len(df) == 0:
        print(f"   ⚠️ Nenhum dado encontrado para {nome}!")
        return pd.DataFrame()
    
    df = df.copy()
    
    # Features temporais básicas
    df['ano'] = df['Data'].dt.year.astype(int)
    df['mes'] = df['Data'].dt.month.astype(int)
    df['dia'] = df['Data'].dt.day.astype(int)
    df['dia_semana'] = df['Data'].dt.dayofweek.astype(int)
    df['trimestre'] = df['Data'].dt.quarter.astype(int)
    
    # Features cíclicas (para capturar sazonalidade)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    
    # Feature: é fim de semana?
    df['fim_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    # Feature: dia do mês (normalizado)
    df['dia_proporcao'] = df['dia'] / 31
    
    # Agrupar por dia
    df_diario = df.groupby('Data').agg({
        'Valor': ['sum', 'count', 'mean'],
        'ano': 'first',
        'mes': 'first',
        'dia': 'first',
        'dia_semana': 'first',
        'trimestre': 'first',
        'mes_sin': 'first',
        'mes_cos': 'first',
        'dia_semana_sin': 'first',
        'dia_semana_cos': 'first',
        'fim_semana': 'first',
        'dia_proporcao': 'first'
    }).reset_index()
    
    # Aplainar colunas
    df_diario.columns = ['Data', f'{nome}_total', f'{nome}_num_transacoes', f'{nome}_media',
                         'ano', 'mes', 'dia', 'dia_semana', 'trimestre',
                         'mes_sin', 'mes_cos', 'dia_semana_sin',
                         'dia_semana_cos', 'fim_semana', 'dia_proporcao']
    
    # Garantir que as colunas são inteiros
    for col in ['ano', 'mes', 'dia', 'dia_semana', 'trimestre', 'fim_semana']:
        if col in df_diario.columns:
            df_diario[col] = df_diario[col].fillna(0).astype(int)
    
    # Preencher dias sem transações com 0
    datas_completas = pd.date_range(start=df_diario['Data'].min(), 
                                    end=df_diario['Data'].max(), 
                                    freq='D')
    df_diario = df_diario.set_index('Data').reindex(datas_completas).reset_index()
    df_diario.rename(columns={'index': 'Data'}, inplace=True)
    df_diario[f'{nome}_total'] = df_diario[f'{nome}_total'].fillna(0)
    df_diario[f'{nome}_num_transacoes'] = df_diario[f'{nome}_num_transacoes'].fillna(0)
    df_diario[f'{nome}_media'] = df_diario[f'{nome}_media'].fillna(0)
    
    # Preencher features temporais para os dias sem dados
    for col in ['ano', 'mes', 'dia', 'dia_semana', 'trimestre', 'mes_sin', 'mes_cos',
                'dia_semana_sin', 'dia_semana_cos', 'fim_semana', 'dia_proporcao']:
        if col in df_diario.columns:
            df_diario[col] = df_diario[col].fillna(method='ffill').fillna(method='bfill')
    
    # Converter novamente para inteiros
    for col in ['ano', 'mes', 'dia', 'dia_semana', 'trimestre', 'fim_semana']:
        if col in df_diario.columns:
            df_diario[col] = df_diario[col].astype(int)
    
    print(f"   ✅ Período: {df_diario['Data'].min().date()} a {df_diario['Data'].max().date()}")
    print(f"   📊 Total de dias: {len(df_diario)}")
    print(f"   📈 {nome} total: R$ {df_diario[f'{nome}_total'].sum():,.2f}")
    
    return df_diario


def dividir_dados_temporais(df, target_col, nome):
    """
    Divide os dados respeitando a ordem temporal
    70% treino, 20% teste, 10% validação
    """
    print(f"\n🔄 Dividindo dados temporais para {nome}...")
    
    if len(df) == 0:
        return None, None, None, None, None, None, None, None, None, None
    
    # Features (X) e Target (y)
    feature_cols = ['ano', 'mes', 'dia', 'dia_semana', 'trimestre',
                    'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos',
                    'fim_semana', 'dia_proporcao']
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Tamanhos
    n = len(df)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    
    # Divisão temporal
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    
    X_test = X.iloc[n_train:n_train + n_test]
    y_test = y.iloc[n_train:n_train + n_test]
    
    X_val = X.iloc[n_train + n_test:]
    y_val = y.iloc[n_train + n_test:]
    
    # Datas para referência
    datas_train = df['Data'].iloc[:n_train]
    datas_test = df['Data'].iloc[n_train:n_train + n_test]
    datas_val = df['Data'].iloc[n_train + n_test:]
    
    print(f"\n   📅 Períodos:")
    print(f"      Treino: {datas_train.min().date()} a {datas_train.max().date()} ({len(X_train)} dias)")
    print(f"      Teste:  {datas_test.min().date()} a {datas_test.max().date()} ({len(X_test)} dias)")
    print(f"      Val:    {datas_val.min().date()} a {datas_val.max().date()} ({len(X_val)} dias)")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, datas_train, datas_test, datas_val, feature_cols


def treinar_modelo(X_train, y_train, X_test, y_test, nome):
    """
    Treina modelos e retorna o melhor
    """
    print(f"\n🤖 Treinando modelo para {nome}...")
    
    # Definição dos modelos com parâmetros robustos
    modelos = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ),
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=1.0, max_iter=10000)
    }
    
    resultados = {}
    
    for nome_modelo, modelo in modelos.items():
        # Treinar
        modelo.fit(X_train, y_train)
        
        # Prever
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        resultados[nome_modelo] = {
            'modelo': modelo,
            'rmse_test': rmse_test,
            'r2_test': r2_test
        }
        
        print(f"   📈 {nome_modelo}: RMSE=R$ {rmse_test:.2f}, R²={r2_test:.4f}")
    
    # Selecionar melhor modelo (baseado no RMSE)
    melhor_nome = min(resultados, key=lambda x: resultados[x]['rmse_test'])
    melhor_modelo = resultados[melhor_nome]['modelo']
    
    print(f"\n🏆 Melhor modelo para {nome}: {melhor_nome}")
    print(f"   RMSE Teste: R$ {resultados[melhor_nome]['rmse_test']:.2f}")
    print(f"   R² Teste: {resultados[melhor_nome]['r2_test']:.4f}")
    
    return melhor_modelo, melhor_nome, resultados


def prever_proximos_meses(modelo, df_original, feature_cols, meses=12):
    """
    Faz previsões para os próximos meses - VERSÃO CORRIGIDA
    """
    print(f"\n🔮 Prevendo para os próximos {meses} meses...")
    
    ultima_data = df_original['Data'].max()
    dias_prever = meses * 30  # Aproximação
    
    datas_futuras = pd.date_range(
        start=ultima_data + timedelta(days=1),
        periods=dias_prever,
        freq='D'
    )
    
    # Criar DataFrame futuro com as features
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
    
    X_futuro = df_futuro[feature_cols]
    
    # Prever
    previsoes = modelo.predict(X_futuro)
    previsoes = np.maximum(previsoes, 0)
    
    df_previsoes = pd.DataFrame({
        'Data': datas_futuras,
        'Valor_Previsto': previsoes
    })
    
    # Agrupar por mês
    df_previsoes['ano'] = df_previsoes['Data'].dt.year
    df_previsoes['mes'] = df_previsoes['Data'].dt.month
    
    resumo_mensal = df_previsoes.groupby(['ano', 'mes']).agg({
        'Valor_Previsto': 'sum'
    }).round(2).reset_index()
    resumo_mensal.columns = ['Ano', 'Mês', 'Total_Previsto']
    
    # CORREÇÃO: Converter para inteiros antes de formatar
    resumo_mensal['Ano'] = resumo_mensal['Ano'].astype(int)
    resumo_mensal['Mês'] = resumo_mensal['Mês'].astype(int)
    
    # Criar nome do mês com formatação segura
    resumo_mensal['Mês_Nome'] = resumo_mensal.apply(
        lambda x: f"{int(x['Mês']):02d}/{int(x['Ano'])}", axis=1
    )
    
    print(f"\n   📈 Resumo das previsões:")
    print(f"      Total previsto: R$ {df_previsoes['Valor_Previsto'].sum():,.2f}")
    print(f"      Média mensal: R$ {resumo_mensal['Total_Previsto'].mean():,.2f}")
    print(f"      Mês com maior previsão: {resumo_mensal.loc[resumo_mensal['Total_Previsto'].idxmax(), 'Mês_Nome']} - R$ {resumo_mensal['Total_Previsto'].max():,.2f}")
    
    return df_previsoes, resumo_mensal


def plotar_resultados(df_historico_receitas, df_historico_despesas, 
                      df_previsao_receitas, df_previsao_despesas,
                      resumo_mensal_receitas, resumo_mensal_despesas):
    """
    Plota gráficos comparativos
    """
    # Verificar se os dados existem
    if len(df_historico_receitas) == 0 or len(df_historico_despesas) == 0:
        print("⚠️ Dados insuficientes para gerar gráficos")
        return None
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # 1. Série histórica e previsão - Receitas
    ax1 = axes[0, 0]
    ax1.plot(df_historico_receitas['Data'], df_historico_receitas['receita_total'], 
             'b-', label='Histórico', linewidth=1.5, alpha=0.7)
    if len(df_previsao_receitas) > 0:
        ax1.plot(df_previsao_receitas['Data'], df_previsao_receitas['Valor_Previsto'], 
                 'r--', label='Previsão', linewidth=2)
    ax1.axvline(x=df_historico_receitas['Data'].max(), color='green', 
                linestyle=':', label='Hoje')
    ax1.set_title('Previsão de Receitas - Próximos 12 Meses', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Receita Diária (R$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Série histórica e previsão - Despesas
    ax2 = axes[0, 1]
    ax2.plot(df_historico_despesas['Data'], df_historico_despesas['despesa_total'], 
             'b-', label='Histórico', linewidth=1.5, alpha=0.7)
    if len(df_previsao_despesas) > 0:
        ax2.plot(df_previsao_despesas['Data'], df_previsao_despesas['Valor_Previsto'], 
                 'r--', label='Previsão', linewidth=2)
    ax2.axvline(x=df_historico_despesas['Data'].max(), color='green', 
                linestyle=':', label='Hoje')
    ax2.set_title('Previsão de Despesas - Próximos 12 Meses', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Despesa Diária (R$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparativo mensal - Barras
    ax3 = axes[1, 0]
    meses = resumo_mensal_receitas['Mês_Nome'].tolist()
    x = np.arange(len(meses))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, resumo_mensal_receitas['Total_Previsto'], 
                    width, label='Receitas', color='green', alpha=0.7)
    bars2 = ax3.bar(x + width/2, resumo_mensal_despesas['Total_Previsto'], 
                    width, label='Despesas', color='red', alpha=0.7)
    
    ax3.set_title('Previsão Mensal - Receitas vs Despesas', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Mês')
    ax3.set_ylabel('Valor (R$)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(meses, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Saldo mensal previsto
    ax4 = axes[1, 1]
    saldo_mensal = (resumo_mensal_receitas['Total_Previsto'].values - 
                    resumo_mensal_despesas['Total_Previsto'].values)
    
    colors = ['green' if s >= 0 else 'red' for s in saldo_mensal]
    ax4.bar(meses, saldo_mensal, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('Saldo Mensal Previsto', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mês')
    ax4.set_ylabel('Saldo (R$)')
    ax4.set_xticklabels(meses, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Saldo acumulado previsto
    ax5 = axes[2, 0]
    saldo_acumulado = np.cumsum(saldo_mensal)
    ax5.fill_between(range(len(meses)), 0, saldo_acumulado, alpha=0.3, color='blue')
    ax5.plot(range(len(meses)), saldo_acumulado, 'b-', linewidth=2)
    ax5.set_title('Saldo Acumulado Previsto', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Mês')
    ax5.set_ylabel('Saldo Acumulado (R$)')
    ax5.set_xticks(range(len(meses)))
    ax5.set_xticklabels(meses, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Resumo métricas
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    total_receitas = resumo_mensal_receitas['Total_Previsto'].sum()
    total_despesas = resumo_mensal_despesas['Total_Previsto'].sum()
    saldo_total = total_receitas - total_despesas
    
    # Encontrar melhor mês
    melhor_mes_idx = np.argmax(saldo_mensal)
    melhor_mes = meses[melhor_mes_idx] if melhor_mes_idx < len(meses) else "N/A"
    
    texto_resumo = f"""
    📊 RESUMO DAS PREVISÕES (12 MESES)
    {'='*40}
    
    RECEITAS TOTAIS:     R$ {total_receitas:,.2f}
    DESPESAS TOTAIS:     R$ {total_despesas:,.2f}
    {'-'*40}
    SALDO TOTAL:         R$ {saldo_total:,.2f}
    
    {'='*40}
    
    📈 MÉDIA MENSAL:
    Receitas:  R$ {total_receitas/12:,.2f}
    Despesas:  R$ {total_despesas/12:,.2f}
    
    💰 MELHOR MÊS (Receitas):
    {resumo_mensal_receitas.loc[resumo_mensal_receitas['Total_Previsto'].idxmax(), 'Mês_Nome']} - R$ {resumo_mensal_receitas['Total_Previsto'].max():,.2f}
    
    📉 MELHOR MÊS (Saldo):
    {melhor_mes} - R$ {saldo_mensal.max():,.2f}
    """
    
    ax6.text(0.1, 0.9, texto_resumo, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    """
    Função principal
    """
    # Caminho do arquivo
    pasta = r"C:\Users\Acer\Desktop\fluxo_caixa_os\Gestao_Preditiva_Fluxo_Caixa_app_Streamlit_TCC2026"
    
    # Listar arquivos Excel disponíveis
    print("\n📂 Arquivos Excel disponíveis:")
    arquivos = [f for f in os.listdir(pasta) if f.endswith('.xlsx')]
    for i, arquivo in enumerate(arquivos):
        print(f"   {i+1}. {arquivo}")
    
    if not arquivos:
        print("❌ Nenhum arquivo Excel encontrado!")
        return
    
    escolha = input("\n📌 Escolha o número do arquivo (ou digite o nome completo): ").strip()
    
    try:
        idx = int(escolha) - 1
        arquivo = arquivos[idx]
    except:
        arquivo = escolha
    
    caminho_completo = os.path.join(pasta, arquivo)
    
    if not os.path.exists(caminho_completo):
        print(f"❌ Arquivo não encontrado: {caminho_completo}")
        return
    
    # 1. Carregar dados
    df_receitas_raw, df_despesas_raw = carregar_dados_completos(caminho_completo)
    
    # 2. Criar features diárias
    df_diario_receitas = criar_features_diarias(df_receitas_raw, 'receita')
    df_diario_despesas = criar_features_diarias(df_despesas_raw, 'despesa')
    
    if len(df_diario_receitas) == 0:
        print("❌ Erro: Não foi possível criar dados diários de receitas!")
        return
    
    if len(df_diario_despesas) == 0:
        print("❌ Erro: Não foi possível criar dados diários de despesas!")
        return
    
    # 3. Dividir dados e treinar modelo de RECEITAS
    print("\n" + "=" * 60)
    print("📈 TREINANDO MODELO DE RECEITAS")
    print("=" * 60)
    
    X_train_rec, X_test_rec, X_val_rec, y_train_rec, y_test_rec, y_val_rec, \
    datas_train_rec, datas_test_rec, datas_val_rec, feature_cols_rec = dividir_dados_temporais(
        df_diario_receitas, 'receita_total', 'RECEITAS')
    
    if X_train_rec is not None:
        scaler_rec = RobustScaler()
        X_train_scaled = scaler_rec.fit_transform(X_train_rec)
        X_test_scaled = scaler_rec.transform(X_test_rec)
        
        modelo_receitas, nome_rec, resultados_rec = treinar_modelo(
            X_train_scaled, y_train_rec, X_test_scaled, y_test_rec, 'RECEITAS')
    else:
        print("❌ Erro ao preparar dados de receitas!")
        return
    
    # 4. Dividir dados e treinar modelo de DESPESAS
    print("\n" + "=" * 60)
    print("📉 TREINANDO MODELO DE DESPESAS")
    print("=" * 60)
    
    X_train_desp, X_test_desp, X_val_desp, y_train_desp, y_test_desp, y_val_desp, \
    datas_train_desp, datas_test_desp, datas_val_desp, feature_cols_desp = dividir_dados_temporais(
        df_diario_despesas, 'despesa_total', 'DESPESAS')
    
    if X_train_desp is not None:
        scaler_desp = RobustScaler()
        X_train_scaled_desp = scaler_desp.fit_transform(X_train_desp)
        X_test_scaled_desp = scaler_desp.transform(X_test_desp)
        
        modelo_despesas, nome_desp, resultados_desp = treinar_modelo(
            X_train_scaled_desp, y_train_desp, X_test_scaled_desp, y_test_desp, 'DESPESAS')
    else:
        print("❌ Erro ao preparar dados de despesas!")
        return
    
    # 5. Fazer previsões para os próximos 12 meses
    print("\n" + "=" * 60)
    print("🔮 GERANDO PREVISÕES")
    print("=" * 60)
    
    # Previsão de receitas
    df_previsao_receitas, resumo_mensal_receitas = prever_proximos_meses(
        modelo_receitas, df_diario_receitas, feature_cols_rec, meses=12)
    
    # Previsão de despesas
    df_previsao_despesas, resumo_mensal_despesas = prever_proximos_meses(
        modelo_despesas, df_diario_despesas, feature_cols_desp, meses=12)
    
    # 6. Salvar modelos e scalers
    print("\n💾 Salvando modelos...")
    
    joblib.dump(modelo_receitas, os.path.join(pasta, 'modelo_previsao_receitas.pkl'))
    joblib.dump(scaler_rec, os.path.join(pasta, 'scaler_receitas.pkl'))
    joblib.dump(modelo_despesas, os.path.join(pasta, 'modelo_previsao_despesas.pkl'))
    joblib.dump(scaler_desp, os.path.join(pasta, 'scaler_despesas.pkl'))
    
    # Salvar features utilizadas
    joblib.dump(feature_cols_rec, os.path.join(pasta, 'feature_cols_receitas.pkl'))
    joblib.dump(feature_cols_desp, os.path.join(pasta, 'feature_cols_despesas.pkl'))
    
    print(f"   ✅ modelo_previsao_receitas.pkl")
    print(f"   ✅ scaler_receitas.pkl")
    print(f"   ✅ modelo_previsao_despesas.pkl")
    print(f"   ✅ scaler_despesas.pkl")
    
    # 7. Salvar previsões em Excel
    nome_base = arquivo.replace('.xlsx', '')
    
    with pd.ExcelWriter(os.path.join(pasta, f'{nome_base}_previsoes_12meses.xlsx')) as writer:
        resumo_mensal_receitas.to_excel(writer, sheet_name='Receitas_Mensal', index=False)
        resumo_mensal_despesas.to_excel(writer, sheet_name='Despesas_Mensal', index=False)
        
        # Criar resumo comparativo
        resumo_comparativo = pd.DataFrame({
            'Mês': resumo_mensal_receitas['Mês_Nome'],
            'Receitas': resumo_mensal_receitas['Total_Previsto'],
            'Despesas': resumo_mensal_despesas['Total_Previsto'],
            'Saldo_Mensal': resumo_mensal_receitas['Total_Previsto'].values - resumo_mensal_despesas['Total_Previsto'].values
        })
        resumo_comparativo['Saldo_Acumulado'] = resumo_comparativo['Saldo_Mensal'].cumsum()
        resumo_comparativo.to_excel(writer, sheet_name='Resumo_Comparativo', index=False)
        
        # Previsões diárias
        df_previsao_receitas.to_excel(writer, sheet_name='Receitas_Diarias', index=False)
        df_previsao_despesas.to_excel(writer, sheet_name='Despesas_Diarias', index=False)
    
    print(f"   ✅ {nome_base}_previsoes_12meses.xlsx")
    
    # 8. Salvar relatório de métricas
    with open(os.path.join(pasta, f'{nome_base}_metricas_modelos.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 RELATÓRIO DE MÉTRICAS DOS MODELOS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Arquivo analisado: {arquivo}\n")
        f.write(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("MODELO DE RECEITAS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Melhor modelo: {nome_rec}\n")
        f.write(f"RMSE Teste: R$ {resultados_rec[nome_rec]['rmse_test']:.2f}\n")
        f.write(f"R² Teste: {resultados_rec[nome_rec]['r2_test']:.4f}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("MODELO DE DESPESAS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Melhor modelo: {nome_desp}\n")
        f.write(f"RMSE Teste: R$ {resultados_desp[nome_desp]['rmse_test']:.2f}\n")
        f.write(f"R² Teste: {resultados_desp[nome_desp]['r2_test']:.4f}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("PREVISÕES PARA 12 MESES\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Receitas: R$ {resumo_mensal_receitas['Total_Previsto'].sum():,.2f}\n")
        f.write(f"Total Despesas: R$ {resumo_mensal_despesas['Total_Previsto'].sum():,.2f}\n")
        f.write(f"Saldo Total: R$ {(resumo_mensal_receitas['Total_Previsto'].sum() - resumo_mensal_despesas['Total_Previsto'].sum()):,.2f}\n")
    
    print(f"   ✅ {nome_base}_metricas_modelos.txt")
    
    # 9. Gerar gráficos
    print("\n📊 Gerando gráficos...")
    
    fig = plotar_resultados(df_diario_receitas, df_diario_despesas,
                            df_previsao_receitas, df_previsao_despesas,
                            resumo_mensal_receitas, resumo_mensal_despesas)
    
    if fig is not None:
        plt.savefig(os.path.join(pasta, f'{nome_base}_previsao_completa.png'), 
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"   ✅ {nome_base}_previsao_completa.png")
    else:
        print("   ⚠️ Não foi possível gerar o gráfico")
    
    # 10. Imprimir resumo final
    print("\n" + "=" * 60)
    print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"\n📁 Arquivos salvos em: {pasta}")
    print(f"\n📈 RESULTADOS DAS PREVISÕES (12 MESES):")
    print(f"   Receitas totais:  R$ {resumo_mensal_receitas['Total_Previsto'].sum():,.2f}")
    print(f"   Despesas totais:  R$ {resumo_mensal_despesas['Total_Previsto'].sum():,.2f}")
    print(f"   Saldo total:      R$ {(resumo_mensal_receitas['Total_Previsto'].sum() - resumo_mensal_despesas['Total_Previsto'].sum()):,.2f}")
    print(f"\n   Média mensal:")
    print(f"   Receitas: R$ {resumo_mensal_receitas['Total_Previsto'].mean():,.2f}")
    print(f"   Despesas: R$ {resumo_mensal_despesas['Total_Previsto'].mean():,.2f}")
    
    print(f"\n📁 Arquivos gerados:")
    print(f"   - modelo_previsao_receitas.pkl")
    print(f"   - scaler_receitas.pkl")
    print(f"   - modelo_previsao_despesas.pkl")
    print(f"   - scaler_despesas.pkl")
    print(f"   - {nome_base}_previsoes_12meses.xlsx")
    print(f"   - {nome_base}_metricas_modelos.txt")
    print(f"   - {nome_base}_previsao_completa.png")


if __name__ == "__main__":
    main()