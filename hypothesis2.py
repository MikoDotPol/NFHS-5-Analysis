# ============================================
# ГИПОТЕЗА 2: Кластеры штатов по социально-обусловленным рискам здоровья детей
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ГИПОТЕЗА 2: Кластеры штатов по социально-обусловленным рискам здоровья детей")
print("=" * 80)

# ============================================
# ШАГ 1: Загрузка данных
# ============================================

print("\n[1] Загрузка данных...")
df = pd.read_csv('Final.csv')
print(f"Загружено записей: {len(df)}")
print(f"Колонок в датасете: {df.shape[1]}")

# ============================================
# ШАГ 2: Создание переменных
# ============================================

print("\n[2] Создание переменных...")

# Анемия у детей (Hg_child_adjust в г/л, порог 110)
if 'Hg_child_adjust' in df.columns:
    df['anemia_binary'] = (df['Hg_child_adjust'] < 110).astype(int)
    anemia_rate = df['anemia_binary'].mean() * 100
    print(f"+ Детская анемия: {anemia_rate:.1f}% детей")

# Полная вакцинация
if 'DPT_full' in df.columns and 'MEASLES_full' in df.columns:
    df['full_vaccination'] = ((df['DPT_full'] == 1) & (df['MEASLES_full'] == 1)).astype(int)
    vac_rate = df['full_vaccination'].mean() * 100
    print(f"+ Вакцинация: {vac_rate:.1f}% детей")

# Высшее образование
if 'Edu_level' in df.columns:
    df['higher_edu'] = (df['Edu_level'] == 3).astype(int)
    edu_rate = df['higher_edu'].mean() * 100
    print(f"+ Высшее образование: {edu_rate:.1f}% женщин")

# Прокси для санитарных условий (электричество)
if 'House_electricity' in df.columns:
    df['improved_sanitation'] = df['House_electricity'].astype(int)
    san_rate = df['improved_sanitation'].mean() * 100
    print(f"+ Электричество: {san_rate:.1f}% домохозяйств")

# Индекс благосостояния
if 'Wealth_Idx_Lb' in df.columns:
    print(f"+ Индекс благосостояния: доступен")

# Штаты
if 'State' in df.columns:
    print(f"+ Штаты: {df['State'].nunique()} уникальных")

# ============================================
# ШАГ 3: Агрегация на уровне штатов
# ============================================

print("\n[3] Агрегация данных на уровне штатов...")

# Собираем переменные для агрегации
agg_vars = ['State']
if 'anemia_binary' in df.columns:
    agg_vars.append('anemia_binary')
if 'full_vaccination' in df.columns:
    agg_vars.append('full_vaccination')
if 'higher_edu' in df.columns:
    agg_vars.append('higher_edu')
if 'improved_sanitation' in df.columns:
    agg_vars.append('improved_sanitation')
if 'Wealth_Idx_Lb' in df.columns:
    agg_vars.append('Wealth_Idx_Lb')

# Агрегация
df_subset = df[agg_vars].copy()
state_stats = df_subset.groupby('State').mean().reset_index()
state_stats['sample_size'] = df.groupby('State').size().values
print(f"Создана агрегация для {len(state_stats)} штатов")

# Переименовываем колонки
rename_dict = {
    'anemia_binary': 'anemia_rate',
    'full_vaccination': 'vaccination_rate',
    'higher_edu': 'higher_edu_rate',
    'improved_sanitation': 'sanitation_rate',
    'Wealth_Idx_Lb': 'wealth_mean'
}
rename_dict = {k: v for k, v in rename_dict.items() if k in state_stats.columns}
if rename_dict:
    state_stats = state_stats.rename(columns=rename_dict)

# Переводим доли в проценты
for col in ['anemia_rate', 'vaccination_rate', 'higher_edu_rate', 'sanitation_rate']:
    if col in state_stats.columns:
        state_stats[col] = state_stats[col] * 100

print("Первые 5 штатов:")
print(state_stats.head().to_string())

# ============================================
# ШАГ 4: Индекс социального благополучия
# ============================================

print("\n[4] Создание индекса социального благополучия...")

# Факторы для индекса
available_factors = []
for col in ['higher_edu_rate', 'sanitation_rate', 'wealth_mean']:
    if col in state_stats.columns:
        available_factors.append(col)

if len(available_factors) >= 2:
    # Нормализация факторов к 0-100
    index_components = []
    for factor in available_factors:
        min_val = state_stats[factor].min()
        max_val = state_stats[factor].max()
        if max_val > min_val:
            normalized = (state_stats[factor] - min_val) / (max_val - min_val) * 100
        else:
            normalized = pd.Series([50] * len(state_stats))
        index_components.append(normalized)
    
    # Индексы благополучия и неблагополучия
    state_stats['wellbeing_index'] = np.mean(index_components, axis=0)
    state_stats['deprivation_index'] = 100 - state_stats['wellbeing_index']
    
    print(f"+ Индексы созданы из {len(available_factors)} факторов")
    
    if 'anemia_rate' in state_stats.columns:
        corr = state_stats['deprivation_index'].corr(state_stats['anemia_rate'])
        print(f"  Корреляция неблагополучия с анемией: {corr:.3f}")

# ============================================
# ШАГ 5: Визуализация распределений
# ============================================

print("\n[5] Визуализация распределений...")

# Гистограммы
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Распределение показателей по штатам Индии', fontsize=16)

# Анемия
if 'anemia_rate' in state_stats.columns:
    ax = axes[0, 0]
    ax.hist(state_stats['anemia_rate'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(state_stats['anemia_rate'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Среднее: {state_stats['anemia_rate'].mean():.1f}%")
    ax.set_xlabel('Доля детей с анемией (%)')
    ax.set_ylabel('Количество штатов')
    ax.set_title('Распределение анемии')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Вакцинация
if 'vaccination_rate' in state_stats.columns:
    ax = axes[0, 1]
    ax.hist(state_stats['vaccination_rate'], bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(state_stats['vaccination_rate'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Среднее: {state_stats['vaccination_rate'].mean():.1f}%")
    ax.set_xlabel('Доля детей с полной вакцинацией (%)')
    ax.set_ylabel('Количество штатов')
    ax.set_title('Распределение вакцинации')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Образование
if 'higher_edu_rate' in state_stats.columns:
    ax = axes[1, 0]
    ax.hist(state_stats['higher_edu_rate'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(state_stats['higher_edu_rate'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Среднее: {state_stats['higher_edu_rate'].mean():.1f}%")
    ax.set_xlabel('Доля женщин с высшим образованием (%)')
    ax.set_ylabel('Количество штатов')
    ax.set_title('Распределение образования')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Благосостояние
if 'wealth_mean' in state_stats.columns:
    ax = axes[1, 1]
    ax.hist(state_stats['wealth_mean'], bins=15, color='gold', edgecolor='black', alpha=0.7)
    ax.axvline(state_stats['wealth_mean'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Среднее: {state_stats['wealth_mean'].mean():.2f}")
    ax.set_xlabel('Индекс благосостояния')
    ax.set_ylabel('Количество штатов')
    ax.set_title('Распределение благосостояния')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Диаграмма рассеяния
if 'anemia_rate' in state_stats.columns and 'deprivation_index' in state_stats.columns:
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(state_stats['deprivation_index'], state_stats['anemia_rate'],
                          c=state_stats['wealth_mean'], s=100, alpha=0.7, cmap='viridis')
    
    # Линия тренда
    z = np.polyfit(state_stats['deprivation_index'], state_stats['anemia_rate'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(state_stats['deprivation_index'].min(), state_stats['deprivation_index'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # Корреляция
    corr, p_val = pearsonr(state_stats['deprivation_index'], state_stats['anemia_rate'])
    plt.text(0.05, 0.95, f'Корреляция: r = {corr:.2f}\np-value = {p_val:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.colorbar(scatter, label='Индекс благосостояния')
    plt.xlabel('Индекс социального неблагополучия')
    plt.ylabel('Доля детей с анемией (%)')
    plt.title('Связь между неблагополучием и анемией')
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================
# ШАГ 6: Корреляционный анализ
# ============================================

print("\n[6] Корреляционный анализ...")

# Матрица корреляций
corr_cols = [col for col in ['anemia_rate', 'vaccination_rate', 'higher_edu_rate', 
                              'sanitation_rate', 'wealth_mean', 'deprivation_index']
             if col in state_stats.columns]

if len(corr_cols) >= 2:
    corr_matrix = state_stats[corr_cols].corr()
    
    print("\nМатрица корреляций:")
    print(corr_matrix.round(3))
    
    # Тепловая карта
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.show()
    
    # Корреляции с анемией
    if 'anemia_rate' in corr_matrix.columns:
        print("\nКорреляции с анемией:")
        anemia_corr = corr_matrix['anemia_rate'].drop('anemia_rate').sort_values(ascending=False)
        for factor, corr in anemia_corr.items():
            print(f"  {factor}: {corr:.3f}")

# ============================================
# ШАГ 7: Кластерный анализ
# ============================================

print("\n[7] Кластерный анализ штатов...")

# Признаки для кластеризации
cluster_features = [col for col in ['anemia_rate', 'vaccination_rate', 'higher_edu_rate', 
                                     'sanitation_rate', 'wealth_mean']
                    if col in state_stats.columns]

if len(cluster_features) >= 2:
    X = state_stats[cluster_features].dropna()
    
    if len(X) > 5:
        # Стандартизация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Определение оптимального числа кластеров
        sil_scores = []
        K_range = range(2, min(6, len(X)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            sil_scores.append(silhouette_score(X_scaled, labels))
        
        # Визуализация метода локтя
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inertia
        inertia = []
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        axes[0].plot(K_range, inertia, 'bo-')
        axes[0].set_xlabel('Количество кластеров')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Метод локтя')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[1].plot(K_range, sil_scores, 'ro-')
        axes[1].set_xlabel('Количество кластеров')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Оценка качества')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Оптимальное число кластеров
        optimal_k = K_range[np.argmax(sil_scores)]
        print(f"\nОптимальное число кластеров: {optimal_k}")
        
        # Финальная кластеризация
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        state_stats['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Статистика по кластерам
        print("\nСтатистика по кластерам (средние):")
        cluster_stats = state_stats.groupby('cluster')[cluster_features].mean().round(1)
        print(cluster_stats.to_string())
        
        # Штаты по кластерам
        print("\nШтаты по кластерам:")
        for cluster_num in sorted(state_stats['cluster'].unique()):
            states = state_stats[state_stats['cluster'] == cluster_num]['State'].tolist()
            print(f"\nКластер {cluster_num} ({len(states)} штатов):")
            print(f"  {', '.join(map(str, states))}")
        
        # Проверка значимости
        print("\n[8] Проверка значимости различий:")
        for factor in cluster_features:
            groups = [state_stats[state_stats['cluster'] == c][factor].dropna() 
                     for c in sorted(state_stats['cluster'].unique())]
            f_stat, p_val = f_oneway(*groups)
            status = "ЗНАЧИМО" if p_val < 0.05 else "НЕ значимо"
            print(f"  {factor}: {status} (p={p_val:.4f})")

# ============================================
# ШАГ 9: Итоговая сводка
# ============================================

print("\n" + "=" * 80)
print("[9] Итоговая сводка по штатам")
print("=" * 80)

if 'anemia_rate' in state_stats.columns:
    # Топ-10 по анемии
    top = state_stats.nlargest(10, 'anemia_rate')[['State', 'anemia_rate']]
    for col in ['vaccination_rate', 'higher_edu_rate', 'wealth_mean', 'deprivation_index']:
        if col in state_stats.columns:
            top[col] = state_stats.loc[top.index, col].round(1)
    
    print("\nТоп-10 штатов с наибольшей долей анемии:")
    print(top.to_string(index=False))
    
    # Топ-10 без анемии
    bottom = state_stats.nsmallest(10, 'anemia_rate')[['State', 'anemia_rate']]
    for col in ['vaccination_rate', 'higher_edu_rate', 'wealth_mean', 'deprivation_index']:
        if col in state_stats.columns:
            bottom[col] = state_stats.loc[bottom.index, col].round(1)
    
    print("\nТоп-10 штатов с наименьшей долей анемии:")
    print(bottom.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)



full_table_columns = ['State', 'anemia_rate', 'vaccination_rate', 'higher_edu_rate', 
                      'sanitation_rate', 'wealth_mean', 'wellbeing_index', 
                      'deprivation_index', 'sample_size', 'cluster']

# Проверяем какие колонки доступны
available_full = [col for col in full_table_columns if col in state_stats.columns]

if len(available_full) > 1:
    # Формируем таблицу
    full_table = state_stats[available_full].round(2)
    
    # Сортируем по названию штата для алфавитного порядка
    full_table = full_table.sort_values('State')
    
    # Переименовываем колонки для читаемости
    column_names_ru = {
        'State': 'Штат',
        'anemia_rate': 'Анемия (%)',
        'vaccination_rate': 'Вакцинация (%)',
        'higher_edu_rate': 'Высшее образование (%)',
        'sanitation_rate': 'Электричество (%)',
        'wealth_mean': 'Благосостояние',
        'wellbeing_index': 'Индекс благополучия',
        'deprivation_index': 'Индекс неблагополучия',
        'sample_size': 'Размер выборки',
        'cluster': 'Кластер'
    }
    
    # Применяем переименование только для существующих колонок
    rename_ru = {k: v for k, v in column_names_ru.items() if k in full_table.columns}
    full_table_ru = full_table.rename(columns=rename_ru)
    
    print("\n" + "=" * 120)
    print("ПОЛНАЯ ТАБЛИЦА ВСЕХ ШТАТОВ ИНДИИ (36 штатов)")
    print("=" * 120)
    
    # Выводим всю таблицу
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.colheader_justify', 'center')
    
    print(full_table_ru.to_string(index=False))
    
    # Сбрасываем настройки pandas
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')