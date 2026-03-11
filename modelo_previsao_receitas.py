# ============================================
# MODELO DE PREVISÃO DE RECEITAS - VERSÃO CORRIGIDA
# Corrigido: problema de features e overfitting
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
from xgboost import XGBRegressor

# Pré-processamento e métricas
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

# Para salvar o modelo
import joblib
import os

# Configurar visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("📊 MODELO DE PREVISÃO DE RECEITAS - VERSÃO CORRIGIDA")
print("=" * 60)

# ============================================
# PASSO 1: CARREGAR E PREPARAR OS DADOS
# ============================================

def carregar_dados(caminho_arquivo):
    """
    Carrega e prepara os dados para modelagem
    """
    print("\n📂 Carregando dados...")
    
    df = pd.read_excel(caminho_arquivo)
    print(f"   ✅ Arquivo carregado: {caminho_arquivo}")
    print(f"   📊 Total de registros: {len(df)}")
    
    # Converter data
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
    
    # Converter valores
    df['Valor'] = pd.to_numeric(df['Valor (R$)'], errors='coerce')
    
    # Filtrar apenas receitas (valores positivos)
    df_receitas = df[df['Valor'] > 0].copy()
    print(f"   💰 Total de receitas: {len(df_receitas)}")
    
    # Remover valores nulos
    df_receitas = df_receitas.dropna(subset=['Data', 'Valor'])
    
    return df_receitas

def criar_features(df):
    """
    Cria features temporais para o modelo (SEM LAGS para evitar data leakage)
    """
    print("\n🔄 Criando features temporais...")
    
    df = df.copy()
    
    # Features temporais básicas
    df['ano'] = df['Data'].dt.year
    df['mes'] = df['Data'].dt.month
    df['dia'] = df['Data'].dt.day
    df['dia_semana'] = df['Data'].dt.dayofweek
    df['trimestre'] = df['Data'].dt.quarter
    
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
    df_diario.columns = ['Data', 'receita_total', 'num_transacoes', 'receita_media', 
                        'ano', 'mes', 'dia', 'dia_semana', 'trimestre',
                        'mes_sin', 'mes_cos', 'dia_semana_sin', 
                        'dia_semana_cos', 'fim_semana', 'dia_proporcao']
    
    print(f"   ✅ Features criadas: {df_diario.shape[1]} colunas")
    print(f"   📅 Período: {df_diario['Data'].min().date()} a {df_diario['Data'].max().date()}")
    print(f"   📊 Total de dias: {len(df_diario)}")
    
    return df_diario

# ============================================
# PASSO 2: DIVISÃO DOS DADOS
# ============================================

def dividir_dados_temporais(df, target_col='receita_total'):
    """
    Divide os dados respeitando a ordem temporal
    70% treino, 20% teste, 10% validação
    """
    print("\n🔄 Dividindo dados temporais...")
    print("   📊 70% Treino | 20% Teste | 10% Validação")
    
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

# ============================================
# PASSO 3: TREINAR MODELOS (COM REGULARIZAÇÃO)
# ============================================

def treinar_modelos(X_train, y_train, X_test, y_test):
    """
    Treina diferentes modelos com regularização para evitar overfitting
    """
    print("\n🤖 Treinando múltiplos modelos com regularização...")
    
    # Definição dos modelos com parâmetros mais robustos
    modelos = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=8,  # Reduzido para evitar overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,  # Usar apenas 80% dos dados em cada árvore
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # Regularização L1
            reg_lambda=1.0,  # Regularização L2
            random_state=42,
            verbosity=0
        ),
        'Ridge': Ridge(alpha=10.0),  # Regularização L2 forte
        'Lasso': Lasso(alpha=1.0, max_iter=10000)  # Regularização L1
    }
    
    resultados = {}
    melhores_modelos = {}
    
    for nome, modelo in modelos.items():
        print(f"\n   📈 Treinando {nome}...")
        
        # Treinar
        modelo.fit(X_train, y_train)
        
        # Prever
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        
        # Métricas
        resultados[nome] = {
            'modelo': modelo,
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mape': mean_absolute_percentage_error(y_train, y_pred_train) * 100,
            'test_mape': mean_absolute_percentage_error(y_test, y_pred_test) * 100
        }
        
        print(f"      ✅ RMSE Treino: R$ {resultados[nome]['train_rmse']:.2f}")
        print(f"      ✅ RMSE Teste:  R$ {resultados[nome]['test_rmse']:.2f}")
        print(f"      ✅ R² Teste:     {resultados[nome]['test_r2']:.4f}")
    
    # Selecionar melhor modelo (baseado no RMSE de teste)
    melhor_modelo_nome = min(resultados, key=lambda x: resultados[x]['test_rmse'])
    melhor_modelo = resultados[melhor_modelo_nome]['modelo']
    
    print(f"\n🏆 Melhor modelo: {melhor_modelo_nome}")
    print(f"   RMSE Teste: R$ {resultados[melhor_modelo_nome]['test_rmse']:.2f}")
    print(f"   R² Teste: {resultados[melhor_modelo_nome]['test_r2']:.4f}")
    
    return melhor_modelo, melhor_modelo_nome, resultados

# ============================================
# PASSO 4: VALIDAÇÃO FINAL
# ============================================

def validar_modelo(modelo, X_val, y_val, datas_val):
    """
    Valida o modelo no conjunto de validação
    """
    print("\n🔍 Validando modelo no conjunto de validação...")
    
    # Previsões
    y_pred_val = modelo.predict(X_val)
    
    # Métricas
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val) * 100
    
    print(f"\n   📊 Métricas de Validação:")
    print(f"      RMSE:  R$ {rmse_val:.2f}")
    print(f"      MAE:   R$ {mae_val:.2f}")
    print(f"      R²:    {r2_val:.4f}")
    print(f"      MAPE:  {mape_val:.2f}%")
    
    # Plotar comparação
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Série temporal
    ax1 = axes[0, 0]
    ax1.plot(datas_val, y_val, 'b-', label='Real', linewidth=2)
    ax1.plot(datas_val, y_pred_val, 'r--', label='Previsto', linewidth=2)
    ax1.set_title('Previsão vs Real - Validação', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Receita (R$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dispersão
    ax2 = axes[0, 1]
    ax2.scatter(y_val, y_pred_val, alpha=0.6)
    ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax2.set_title('Valores Reais vs Previstos', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Real (R$)')
    ax2.set_ylabel('Previsto (R$)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Resíduos
    ax3 = axes[1, 0]
    residuos = y_val - y_pred_val
    ax3.scatter(y_pred_val, residuos, alpha=0.6)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('Resíduos vs Valores Previstos', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Previsto (R$)')
    ax3.set_ylabel('Resíduo (R$)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribuição dos erros
    ax4 = axes[1, 1]
    ax4.hist(residuos, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--')
    ax4.set_title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Resíduo (R$)')
    ax4.set_ylabel('Frequência')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validacao_modelo.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return {
        'rmse': rmse_val,
        'mae': mae_val,
        'r2': r2_val,
        'mape': mape_val,
        'residuos': residuos
    }

# ============================================
# PASSO 5: PREVER PRÓXIMO ANO (CORRIGIDO)
# ============================================

def prever_proximo_ano(modelo, df_original, feature_cols, anos=1):
    """
    Faz previsões para o próximo ano (VERSÃO CORRIGIDA)
    """
    print(f"\n🔮 Prevendo receitas para os próximos {anos} ano(s)...")
    
    ultima_data = df_original['Data'].max()
    datas_futuras = pd.date_range(
        start=ultima_data + timedelta(days=1),
        periods=365 * anos,
        freq='D'
    )
    
    # Criar DataFrame futuro com as features corretas
    df_futuro = pd.DataFrame({'Data': datas_futuras})
    
    # Criar TODAS as features que o modelo espera
    df_futuro['ano'] = df_futuro['Data'].dt.year
    df_futuro['mes'] = df_futuro['Data'].dt.month
    df_futuro['dia'] = df_futuro['Data'].dt.day
    df_futuro['dia_semana'] = df_futuro['Data'].dt.dayofweek
    df_futuro['trimestre'] = df_futuro['Data'].dt.quarter
    
    # Features cíclicas
    df_futuro['mes_sin'] = np.sin(2 * np.pi * df_futuro['mes'] / 12)
    df_futuro['mes_cos'] = np.cos(2 * np.pi * df_futuro['mes'] / 12)
    df_futuro['dia_semana_sin'] = np.sin(2 * np.pi * df_futuro['dia_semana'] / 7)
    df_futuro['dia_semana_cos'] = np.cos(2 * np.pi * df_futuro['dia_semana'] / 7)
    
    # Fim de semana
    df_futuro['fim_semana'] = df_futuro['dia_semana'].isin([5, 6]).astype(int)
    
    # Dia proporção
    df_futuro['dia_proporcao'] = df_futuro['dia'] / 31
    
    # Selecionar apenas as features usadas no treinamento
    X_futuro = df_futuro[feature_cols]
    
    # Prever
    previsoes = modelo.predict(X_futuro)
    previsoes = np.maximum(previsoes, 0)  # Garantir valores não negativos
    
    # Criar DataFrame de previsões
    df_previsoes = pd.DataFrame({
        'Data': datas_futuras,
        'Receita_Prevista': previsoes,
        'Receita_Acumulada': np.cumsum(previsoes)
    })
    
    # Adicionar colunas para análise
    df_previsoes['mes'] = df_previsoes['Data'].dt.month
    df_previsoes['ano'] = df_previsoes['Data'].dt.year
    df_previsoes['dia_semana'] = df_previsoes['Data'].dt.dayofweek
    
    # Resumo mensal
    resumo_mensal = df_previsoes.groupby(['ano', 'mes']).agg({
        'Receita_Prevista': ['sum', 'mean', 'count', 'std']
    }).round(2)
    resumo_mensal.columns = ['Total_Mensal', 'Media_Diaria', 'Dias', 'Desvio_Padrao']
    
    print(f"\n   📈 Resumo das previsões:")
    print(f"      Total previsto: R$ {df_previsoes['Receita_Prevista'].sum():,.2f}")
    print(f"      Média diária: R$ {df_previsoes['Receita_Prevista'].mean():.2f}")
    print(f"      Maior dia: R$ {df_previsoes['Receita_Prevista'].max():.2f}")
    
    return df_previsoes, resumo_mensal

# ============================================
# PASSO 6: ANÁLISE DE FEATURES IMPORTANTES
# ============================================

def analisar_importancia_features(modelo, feature_cols, nome_modelo):
    """
    Analisa quais features são mais importantes para o modelo
    """
    print("\n📊 Analisando importância das features...")
    
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        indices = np.argsort(importancias)[::-1]
        
        print("\n   Top 5 features mais importantes:")
        for i in range(min(5, len(feature_cols))):
            print(f"      {i+1}. {feature_cols[indices[i]]}: {importancias[indices[i]]:.3f}")
        
        # Plotar
        plt.figure(figsize=(10, 6))
        plt.title(f'Importância das Features - {nome_modelo}')
        plt.barh(range(len(importancias)), importancias[indices])
        plt.yticks(range(len(importancias)), [feature_cols[i] for i in indices])
        plt.xlabel('Importância')
        plt.tight_layout()
        plt.savefig('importancia_features.png', dpi=100, bbox_inches='tight')
        plt.show()
    else:
        print("   Modelo não suporta análise de importância de features")

# ============================================
# PASSO 7: EXECUÇÃO PRINCIPAL
# ============================================

def main():
    """
    Função principal do pipeline
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
        # Se for número
        idx = int(escolha) - 1
        arquivo = arquivos[idx]
    except:
        # Se for nome
        arquivo = escolha
    
    caminho_completo = os.path.join(pasta, arquivo)
    
    if not os.path.exists(caminho_completo):
        print(f"❌ Arquivo não encontrado: {caminho_completo}")
        return
    
    # 1. Carregar dados
    df_receitas = carregar_dados(caminho_completo)
    
    # 2. Criar features
    df_diario = criar_features(df_receitas)
    
    # 3. Dividir dados
    X_train, X_test, X_val, y_train, y_test, y_val, datas_train, datas_test, datas_val, feature_cols = dividir_dados_temporais(df_diario)
    
    # 4. Escalar features (opcional, mas ajuda modelos lineares)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    # 5. Treinar modelos
    melhor_modelo, nome_melhor, resultados = treinar_modelos(
        X_train_scaled, y_train, 
        X_test_scaled, y_test
    )
    
    # 6. Validar no conjunto de validação
    metricas_val = validar_modelo(melhor_modelo, X_val_scaled, y_val, datas_val)
    
    # 7. Analisar importância das features (se aplicável)
    analisar_importancia_features(melhor_modelo, feature_cols, nome_melhor)
    
    # 8. Prever próximo ano
    df_previsoes, resumo_mensal = prever_proximo_ano(melhor_modelo, df_diario, feature_cols)
    
    # 9. Salvar resultados
    print("\n💾 Salvando resultados...")
    
    # Salvar modelo e scaler
    joblib.dump(melhor_modelo, os.path.join(pasta, 'modelo_previsao_receitas.pkl'))
    joblib.dump(scaler, os.path.join(pasta, 'scaler.pkl'))
    
    # Salvar previsões
    nome_base = arquivo.replace('.xlsx', '')
    df_previsoes.to_excel(os.path.join(pasta, f'{nome_base}_previsoes_proximo_ano.xlsx'), index=False)
    resumo_mensal.to_excel(os.path.join(pasta, f'{nome_base}_resumo_mensal.xlsx'))
    
    # Salvar relatório de métricas
    with open(os.path.join(pasta, f'{nome_base}_metricas.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("📊 RELATÓRIO DE MÉTRICAS DO MODELO\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Arquivo analisado: {arquivo}\n")
        f.write(f"Melhor modelo: {nome_melhor}\n\n")
        
        f.write("Métricas de Treino:\n")
        f.write(f"  RMSE: R$ {resultados[nome_melhor]['train_rmse']:.2f}\n")
        f.write(f"  R²:   {resultados[nome_melhor]['train_r2']:.4f}\n")
        f.write(f"  MAPE: {resultados[nome_melhor]['train_mape']:.2f}%\n\n")
        
        f.write("Métricas de Teste:\n")
        f.write(f"  RMSE: R$ {resultados[nome_melhor]['test_rmse']:.2f}\n")
        f.write(f"  R²:   {resultados[nome_melhor]['test_r2']:.4f}\n")
        f.write(f"  MAPE: {resultados[nome_melhor]['test_mape']:.2f}%\n\n")
        
        f.write("Métricas de Validação:\n")
        f.write(f"  RMSE: R$ {metricas_val['rmse']:.2f}\n")
        f.write(f"  MAE:  R$ {metricas_val['mae']:.2f}\n")
        f.write(f"  R²:   {metricas_val['r2']:.4f}\n")
        f.write(f"  MAPE: {metricas_val['mape']:.2f}%\n\n")
        
        f.write("Previsão para o próximo ano:\n")
        f.write(f"  Total: R$ {df_previsoes['Receita_Prevista'].sum():,.2f}\n")
        f.write(f"  Média mensal: R$ {df_previsoes.groupby('mes')['Receita_Prevista'].sum().mean():,.2f}\n")
    
    # 10. Plotar previsão final
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Gráfico 1: Série histórica + previsão
    ax1.plot(df_diario['Data'], df_diario['receita_total'], 'b-', label='Histórico', linewidth=1.5, alpha=0.7)
    ax1.plot(df_previsoes['Data'], df_previsoes['Receita_Prevista'], 'r--', label='Previsão', linewidth=2)
    ax1.axvline(x=df_diario['Data'].max(), color='green', linestyle=':', label='Hoje')
    ax1.set_title(f'Previsão de Receitas - {nome_melhor}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Receita Diária (R$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Previsão acumulada
    ax2.fill_between(df_previsoes['Data'], 0, df_previsoes['Receita_Acumulada'], alpha=0.3, color='green')
    ax2.plot(df_previsoes['Data'], df_previsoes['Receita_Acumulada'], 'g-', linewidth=2)
    ax2.set_title('Receita Acumulada Prevista', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Receita Acumulada (R$)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, f'{nome_base}_previsao_completa.png'), dpi=100, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 50)
    print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 50)
    print(f"\n📁 Arquivos salvos em: {pasta}")
    print(f"   - modelo_previsao_receitas.pkl")
    print(f"   - scaler.pkl")
    print(f"   - {nome_base}_previsoes_proximo_ano.xlsx")
    print(f"   - {nome_base}_resumo_mensal.xlsx")
    print(f"   - {nome_base}_metricas.txt")
    print(f"   - {nome_base}_previsao_completa.png")
    print(f"   - validacao_modelo.png")
    print(f"   - importancia_features.png")

if __name__ == "__main__":
    main()