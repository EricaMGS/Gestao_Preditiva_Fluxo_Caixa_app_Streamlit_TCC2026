import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Configurar seed para reprodutibilidade
random.seed(42)
np.random.seed(42)

print("=" * 50)
print("📊 ESTIMADOR DE FLUXO DE CAIXA 2024-2025")
print("=" * 50)

# ============================================
# PASSO 1: SOLICITAR O ARQUIVO DE ENTRADA
# ============================================

# Caminho fixo para sua pasta
pasta_destino = r"C:\Users\Acer\Desktop\fluxo_caixa_os\Gestao_Preditiva_Fluxo_Caixa_app_Streamlit_TCC2026"

# Verificar se a pasta existe
if not os.path.exists(pasta_destino):
    print(f"❌ Pasta não encontrada: {pasta_destino}")
    print("Criando a pasta...")
    os.makedirs(pasta_destino, exist_ok=True)

# Nome do arquivo de entrada (assumindo que está na mesma pasta)
arquivo_entrada = os.path.join(pasta_destino, "Extrato_Lançamentos teste1.xlsx")

# Se o arquivo não existir, pedir para o usuário
if not os.path.exists(arquivo_entrada):
    print(f"\n⚠️ Arquivo não encontrado: {arquivo_entrada}")
    arquivo_entrada = input("📂 Digite o caminho completo do seu arquivo Excel: ").strip('"').strip("'")
    
    if not os.path.exists(arquivo_entrada):
        print("❌ Arquivo não encontrado! Usando dados de exemplo...")
        # Usar os dados que você forneceu como exemplo
        dados_exemplo = [
            ['25/01/2026', 'SALDO ANTERIOR', ''],
            ['27/01/2026', 'PIX RECEBIDO', 72],
            ['27/01/2026', 'PIX RECEBIDO', 48],
            ['27/01/2026', 'PIX RECEBIDO', 20.4],
            ['27/01/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['28/01/2026', 'PIX RECEBIDO', 12],
            ['28/01/2026', 'PIX RECEBIDO', 12],
            ['28/01/2026', 'PIX RECEBIDO', 21.6],
            ['28/01/2026', 'PIX RECEBIDO', 43.2],
            ['28/01/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['29/01/2026', 'PIX RECEBIDO', 420],
            ['29/01/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['30/01/2026', 'PIX RECEBIDO', 24],
            ['30/01/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['02/02/2026', 'Aluguel', -1800],
            ['02/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['03/02/2026', 'PIX RECEBIDO', 42],
            ['03/02/2026', 'PIX RECEBIDO', 8.4],
            ['03/02/2026', 'Tarifa conta', -95.4],
            ['03/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['04/02/2026', 'PIX RECEBIDO', 18.012],
            ['04/02/2026', 'PIX RECEBIDO', 87.48],
            ['04/02/2026', 'PIX RECEBIDO', 57.6],
            ['04/02/2026', 'PIX RECEBIDO', 18],
            ['04/02/2026', 'DEPOSITO', 1080],
            ['04/02/2026', 'DEPOSITO', 240],
            ['04/02/2026', 'PIX RECEBIDO', 12],
            ['04/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['06/02/2026', 'PIX RECEBIDO', 14.4],
            ['06/02/2026', 'Água', -177.936],
            ['06/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['09/02/2026', 'PIX RECEBIDO', 60],
            ['09/02/2026', 'PIX RECEBIDO', 72],
            ['09/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['10/02/2026', 'PIX RECEBIDO', 24],
            ['10/02/2026', 'PIX RECEBIDO', 24],
            ['10/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['11/02/2026', 'PIX RECEBIDO', 420],
            ['11/02/2026', 'PIX RECEBIDO', 18],
            ['11/02/2026', 'PIX RECEBIDO', 36],
            ['11/02/2026', 'PIX RECEBIDO', 12],
            ['11/02/2026', 'PIX RECEBIDO', 12],
            ['11/02/2026', 'PIX RECEBIDO', 240],
            ['11/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['18/02/2026', 'PIX RECEBIDO', 14.4],
            ['18/02/2026', 'PIX RECEBIDO', 19.2],
            ['18/02/2026', 'PIX RECEBIDO', 6],
            ['18/02/2026', 'PIX RECEBIDO', 24],
            ['18/02/2026', 'Luz', -133.584],
            ['18/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['23/02/2026', 'PIX RECEBIDO', 36],
            ['23/02/2026', 'PIX RECEBIDO', 84],
            ['23/02/2026', 'Telefone', -148.908],
            ['23/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['24/02/2026', 'PIX RECEBIDO', 18],
            ['24/02/2026', 'PIX RECEBIDO', 21.6],
            ['24/02/2026', 'PIX RECEBIDO', 36],
            ['24/02/2026', 'PIX RECEBIDO', 24],
            ['24/02/2026', 'SALDO TOTAL DISPONÍVEL DIA', ''],
            ['25/02/2026', 'PIX RECEBIDO', 31.2],
            ['25/02/2026', 'PIX RECEBIDO', 33.6],
            ['25/02/2026', 'PIX RECEBIDO', 18],
            ['25/02/2026', 'PIX RECEBIDO', 15.6],
            ['25/02/2026', 'SALDO EM CONTA CORRENTE', '']
        ]
        df_original = pd.DataFrame(dados_exemplo, columns=['Data', 'Lançamento', 'Valor (R$)'])
        print("✅ Usando dados de exemplo fornecidos")
    else:
        # Carregar arquivo fornecido pelo usuário
        df_original = pd.read_excel(arquivo_entrada)
        print(f"✅ Arquivo carregado: {arquivo_entrada}")
else:
    # Carregar arquivo da pasta padrão
    df_original = pd.read_excel(arquivo_entrada)
    print(f"✅ Arquivo carregado: {arquivo_entrada}")

# ============================================
# PASSO 2: PROCESSAR DADOS ORIGINAIS
# ============================================

print("\n📊 Analisando padrões dos dados...")

# Converter data e valores
df_original['Data'] = pd.to_datetime(df_original['Data'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
df_original['Valor'] = pd.to_numeric(df_original['Valor (R$)'], errors='coerce')

# Identificar tipos de lançamentos
receitas_tipos = ['PIX RECEBIDO', 'DEPOSITO']
despesas_tipos = ['Aluguel', 'Tarifa conta', 'Água', 'Luz', 'Telefone']

# Extrair receitas
receitas = df_original[df_original['Lançamento'].isin(receitas_tipos)]['Valor'].dropna()

# Estatísticas de receitas
pix_pequenos = receitas[receitas < 100]
pix_grandes = receitas[receitas >= 100]

media_pix_pequeno = pix_pequenos.mean() if len(pix_pequenos) > 0 else 30
desvio_pix = pix_pequenos.std() if len(pix_pequenos) > 1 else media_pix_pequeno * 0.3

# Frequência de PIX
dias_unicos = len(df_original['Data'].dt.date.unique())
frequencia_pix_dia = len(pix_pequenos) / max(dias_unicos, 1)

# Valores grandes típicos
valores_grandes_tipicos = [240, 420, 1080, 540, 720]
if len(pix_grandes) > 0:
    valores_grandes_tipicos = sorted(set(pix_grandes.round(-1).tolist() + valores_grandes_tipicos))

# Despesas fixas (últimos valores de cada tipo)
despesas_fixas = {}
for despesa in despesas_tipos:
    ultima = df_original[df_original['Lançamento'] == despesa]['Valor'].dropna()
    if len(ultima) > 0:
        despesas_fixas[despesa] = abs(ultima.iloc[-1])
    else:
        # Valores padrão
        valores_padrao = {'Aluguel': 1800, 'Tarifa conta': 95.4, 'Água': 177.94, 'Luz': 133.58, 'Telefone': 148.91}
        despesas_fixas[despesa] = valores_padrao.get(despesa, 100)

print(f"\n📈 Padrões identificados:")
print(f"   - Média PIX pequenos: R$ {media_pix_pequeno:.2f}")
print(f"   - Frequência PIX/dia: {frequencia_pix_dia:.2f}")
print(f"   - Valores grandes típicos: {valores_grandes_tipicos}")

# ============================================
# PASSO 3: FUNÇÃO PARA GERAR ESTIMATIVAS
# ============================================

def gerar_ano_estimado(ano, inflacao=0):
    """Gera estimativas para um ano completo"""
    estimativas = []
    
    for mes in range(1, 13):
        # Determinar dias no mês
        if mes in [1, 3, 5, 7, 8, 10, 12]:
            dias_no_mes = 31
        elif mes in [4, 6, 9, 11]:
            dias_no_mes = 30
        else:  # Fevereiro
            dias_no_mes = 28 if ano % 4 != 0 else 29
        
        # ===== DESPESAS FIXAS =====
        # Aluguel (início do mês)
        data_aluguel = datetime(ano, mes, min(2, dias_no_mes))
        estimativas.append({
            'Data': data_aluguel.strftime('%d/%m/%Y'),
            'Lançamento': 'Aluguel',
            'Valor (R$)': -despesas_fixas.get('Aluguel', 1800) * (1 + inflacao)
        })
        
        # Tarifa conta (dia 03)
        data_tarifa = datetime(ano, mes, min(3, dias_no_mes))
        estimativas.append({
            'Data': data_tarifa.strftime('%d/%m/%Y'),
            'Lançamento': 'Tarifa conta',
            'Valor (R$)': -despesas_fixas.get('Tarifa conta', 95.4) * (1 + inflacao)
        })
        
        # Água (dia 06)
        data_agua = datetime(ano, mes, min(6, dias_no_mes))
        valor_agua = -despesas_fixas.get('Água', 177.94) * (1 + inflacao) * random.uniform(0.9, 1.1)
        estimativas.append({
            'Data': data_agua.strftime('%d/%m/%Y'),
            'Lançamento': 'Água',
            'Valor (R$)': round(valor_agua, 2)
        })
        
        # Luz (dia 18)
        data_luz = datetime(ano, mes, min(18, dias_no_mes))
        valor_luz = -despesas_fixas.get('Luz', 133.58) * (1 + inflacao) * random.uniform(0.85, 1.15)
        estimativas.append({
            'Data': data_luz.strftime('%d/%m/%Y'),
            'Lançamento': 'Luz',
            'Valor (R$)': round(valor_luz, 2)
        })
        
        # Telefone (dia 23)
        data_tel = datetime(ano, mes, min(23, dias_no_mes))
        estimativas.append({
            'Data': data_tel.strftime('%d/%m/%Y'),
            'Lançamento': 'Telefone',
            'Valor (R$)': -despesas_fixas.get('Telefone', 148.91) * (1 + inflacao)
        })
        
        # ===== RECEITAS =====
        # PIX pequenos
        num_pix = max(1, int(np.random.poisson(frequencia_pix_dia * dias_no_mes)))
        
        for _ in range(num_pix):
            dia = random.randint(1, dias_no_mes)
            data_pix = datetime(ano, mes, dia)
            
            valor_base = media_pix_pequeno * (1 + inflacao)
            valor = max(5, np.random.normal(valor_base, desvio_pix * 0.8))
            valor = round(valor * 2) / 2  # Arredondar para 0.5
            
            estimativas.append({
                'Data': data_pix.strftime('%d/%m/%Y'),
                'Lançamento': 'PIX RECEBIDO',
                'Valor (R$)': valor
            })
        
        # Depósitos grandes (1-3 por mês)
        num_depositos = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        for _ in range(num_depositos):
            dia = random.randint(1, dias_no_mes)
            data_dep = datetime(ano, mes, dia)
            
            valor = random.choice(valores_grandes_tipicos) * (1 + inflacao)
            
            estimativas.append({
                'Data': data_dep.strftime('%d/%m/%Y'),
                'Lançamento': 'DEPOSITO',
                'Valor (R$)': valor
            })
        
        # Adicionar SALDOS nos dias com movimento
        dias_mes = set()
        for est in estimativas:
            if est['Data'][3:5] == f'{mes:02d}' and est['Data'][6:] == str(ano) and est['Valor (R$)'] != '':
                dias_mes.add(est['Data'])
        
        for dia_str in sorted(dias_mes):
            tem_saldo = any(e['Data'] == dia_str and 'SALDO' in e['Lançamento'] for e in estimativas)
            if not tem_saldo:
                estimativas.append({
                    'Data': dia_str,
                    'Lançamento': 'SALDO TOTAL DISPONÍVEL DIA',
                    'Valor (R$)': ''
                })
    
    # Ordenar por data
    estimativas.sort(key=lambda x: datetime.strptime(x['Data'], '%d/%m/%Y'))
    return estimativas

# ============================================
# PASSO 4: GERAR ESTIMATIVAS
# ============================================

print("\n🔄 Gerando estimativas para 2024 e 2025...")

# Gerar estimativas
estimativas_2024 = gerar_ano_estimado(2024, inflacao=0)
estimativas_2025 = gerar_ano_estimado(2025, inflacao=0.05)

# Combinar tudo
print("✅ Estimativas geradas com sucesso!")

# Perguntar se quer incluir os dados originais
incluir_originais = input("\n❓ Incluir dados originais de 2026 no arquivo? (s/n): ").lower() == 's'

if incluir_originais:
    # Converter dados originais para o formato
    dados_originais_formatados = []
    for _, row in df_original.iterrows():
        dados_originais_formatados.append({
            'Data': row['Data'].strftime('%d/%m/%Y'),
            'Lançamento': row['Lançamento'],
            'Valor (R$)': row['Valor (R$)'] if pd.notna(row['Valor (R$)']) else ''
        })
    todas_estimativas = estimativas_2024 + estimativas_2025 + dados_originais_formatados
    print("📦 Incluindo dados originais de 2026")
else:
    todas_estimativas = estimativas_2024 + estimativas_2025

# Criar DataFrame final
df_final = pd.DataFrame(todas_estimativas)

# ============================================
# PASSO 5: SALVAR ARQUIVO
# ============================================

# Nome do arquivo de saída
nome_arquivo = f"fluxo_caixa_estimado_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
caminho_completo = os.path.join(pasta_destino, nome_arquivo)

# Salvar Excel
df_final.to_excel(caminho_completo, index=False, sheet_name='Fluxo de Caixa')

print(f"\n✅ ARQUIVO SALVO COM SUCESSO!")
print(f"📁 Local: {caminho_completo}")
print(f"📊 Total de lançamentos: {len(df_final)}")
print(f"📅 Período: 2024 a {2026 if incluir_originais else 2025}")

# Estatísticas rápidas
df_final['Valor_num'] = pd.to_numeric(df_final['Valor (R$)'], errors='coerce')
receitas_total = df_final[df_final['Valor_num'] > 0]['Valor_num'].sum()
despesas_total = abs(df_final[df_final['Valor_num'] < 0]['Valor_num'].sum())

print(f"\n💰 RESUMO FINANCEIRO:")
print(f"   Receitas totais: R$ {receitas_total:,.2f}")
print(f"   Despesas totais: R$ {despesas_total:,.2f}")
print(f"   Saldo projetado: R$ {receitas_total - despesas_total:,.2f}")

print(f"\n📂 Arquivo salvo em: {caminho_completo}")