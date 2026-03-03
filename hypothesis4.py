# ============================================
# ГИПОТЕЗА 4: Индекс уязвимости домохозяйства
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ГИПОТЕЗА 4: Индекс уязвимости домохозяйства")
print("=" * 80)

# ============================================
# ШАГ 1: Загрузка данных
# ============================================

print("\n[1] Загрузка данных...")
df = pd.read_csv('Final.csv')
print(f"Загружено записей: {len(df)}")
print(f"Колонок в датасете: {df.shape[1]}")

# ============================================
# ШАГ 2: Создание целевых переменных (негативные исходы)
# ============================================

print("\n[2] Создание целевых переменных...")

# 1. Анемия у матери (Anemia_level: 0=нет анемии, 1+=есть анемия)
df['maternal_anemia'] = (df['Anemia_level'] > 0).astype(int)
print(f"Анемия у матери: {df['maternal_anemia'].mean():.2%}")

# 2. Анемия у ребенка (Hg_child_adjust в г/л, порог 110 г/л)
df['child_anemia'] = (df['Hg_child_adjust'] < 110).astype(int)
print(f"Анемия у ребенка (Hg_child_adjust < 110 г/л): {df['child_anemia'].mean():.2%}")

# 3. Домашние роды (не в учреждении)
df['home_delivery'] = ((df['DeliveryPlace_Other'] == 0) & (df['DeliveryPlace_Private'] == 0)).astype(int)
print(f"Домашние роды: {df['home_delivery'].mean():.2%}")

# 4. Неполная вакцинация ребенка
df['incomplete_vaccination'] = ((df['DPT_full'] == 0) | (df['MEASLES_full'] == 0)).astype(int)
print(f"Неполная вакцинация: {df['incomplete_vaccination'].mean():.2%}")

# Список негативных исходов
negative_outcomes = [
    ('maternal_anemia', 'Анемия у матери'),
    ('child_anemia', 'Анемия у ребенка'),
    ('home_delivery', 'Домашние роды'),
    ('incomplete_vaccination', 'Неполная вакцинация')
]

print(f"\nВсего негативных исходов для анализа: {len(negative_outcomes)}")

# ============================================
# ШАГ 3: Создание компонентов индекса уязвимости
# ============================================

print("\n" + "=" * 80)
print("[3] Метод 1: Построение индекса уязвимости")
print("=" * 80)

# Компонент 1: Экономическая уязвимость (обратный Wealth_Idx_Lb)
max_wealth = df['Wealth_Idx_Lb'].max()
df['economic_vulnerability'] = max_wealth - df['Wealth_Idx_Lb'] + 1
print(f"Экономическая уязвимость создана")
print(f"  Диапазон: {df['economic_vulnerability'].min()}-{df['economic_vulnerability'].max()}")

# Компонент 2: Образовательная уязвимость (обратный Edu_level)
max_edu = df['Edu_level'].max()
df['education_vulnerability'] = max_edu - df['Edu_level']
print(f"Образовательная уязвимость создана")
print(f"  Диапазон: {df['education_vulnerability'].min()}-{df['education_vulnerability'].max()}")

# Компонент 3: Демографическая нагрузка (количество детей до 5 лет)
df['demographic_vulnerability'] = df['Child_under5'].clip(upper=5)
print(f"Демографическая нагрузка создана")
print(f"  Диапазон: {df['demographic_vulnerability'].min()}-{df['demographic_vulnerability'].max()}")

# Компонент 4: Сельская уязвимость
df['rural_vulnerability'] = 1 - df['ResidenceType_Urban']
print(f"Сельская уязвимость создана")
print(f"  Доля сельских: {df['rural_vulnerability'].mean():.2%}")

# Компонент 5: ИНФРАСТРУКТУРНАЯ УЯЗВИМОСТЬ
print("\n--- Создание инфраструктурного компонента ---")

# Санитария (туалет) - предполагаем что 1-3 это улучшенные туалеты
df['sanitation_vulnerability'] = (~df['Toilet_Facility'].isin([1, 2, 3])).astype(int)
print(f"Санитарная уязвимость: {df['sanitation_vulnerability'].mean():.2%}")

# Источник воды
df['water_vulnerability'] = 1 - df['Water_Source_Piped']
print(f"Водная уязвимость: {df['water_vulnerability'].mean():.2%}")

# Электричество
df['electricity_vulnerability'] = 1 - df['House_electricity']
print(f"Электрическая уязвимость: {df['electricity_vulnerability'].mean():.2%}")

# Создаем композитный инфраструктурный индекс
infrastructure_components = ['sanitation_vulnerability', 'water_vulnerability', 'electricity_vulnerability']
infra_df = df[infrastructure_components].copy()
infra_scaler = MinMaxScaler()
infra_normalized = infra_scaler.fit_transform(infra_df)
df['infrastructure_vulnerability'] = infra_normalized.mean(axis=1)
print(f"Инфраструктурная уязвимость создана")
print(f"  Среднее: {df['infrastructure_vulnerability'].mean():.3f}")

# ============================================
# ШАГ 4: Нормализация компонентов и создание индекса
# ============================================

print("\n[4] Нормализация компонентов и создание индекса...")

# Собираем все компоненты
vulnerability_components = [
    'economic_vulnerability',
    'education_vulnerability', 
    'demographic_vulnerability',
    'rural_vulnerability',
    'infrastructure_vulnerability'
]

component_names = ['Экономическая', 'Образовательная', 'Демографическая', 'Сельская', 'Инфраструктурная']

print(f"Компоненты для индекса: {component_names}")

# Создаем DataFrame с компонентами
components_df = df[vulnerability_components].copy()
components_df = components_df.dropna()
print(f"\nДанных после удаления пропусков: {len(components_df)}")

# Нормализация
scaler = MinMaxScaler()
components_normalized = scaler.fit_transform(components_df)
components_normalized = pd.DataFrame(components_normalized, 
                                     columns=components_df.columns,
                                     index=components_df.index)

print("\nСтатистика после нормализации:")
for col in components_normalized.columns:
    print(f"  {col}: среднее={components_normalized[col].mean():.3f}, "
          f"стд={components_normalized[col].std():.3f}")

# ============================================
# ПУНКТ 5: Корреляционная матрица компонентов
# ============================================
print("\n--- Корреляция между компонентами индекса ---")
corr_matrix = components_normalized.corr()
print(corr_matrix.round(3))

# Визуализация корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            xticklabels=component_names, yticklabels=component_names,
            fmt='.2f', linewidths=0.5)
plt.title('Корреляция между компонентами уязвимости', fontsize=14)
plt.tight_layout()
plt.show()

# Создание индекса
df['vulnerability_index'] = components_normalized.sum(axis=1)
df['vulnerability_index_norm'] = (df['vulnerability_index'] - df['vulnerability_index'].min()) / (df['vulnerability_index'].max() - df['vulnerability_index'].min())

print(f"\nИндекс уязвимости создан!")
print(f"  Диапазон: {df['vulnerability_index_norm'].min():.3f} - {df['vulnerability_index_norm'].max():.3f}")
print(f"  Среднее: {df['vulnerability_index_norm'].mean():.3f}")

# Визуализация распределения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['vulnerability_index_norm'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(df['vulnerability_index_norm'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Среднее: {df['vulnerability_index_norm'].mean():.2f}")
axes[0].set_xlabel('Индекс уязвимости (нормализованный)')
axes[0].set_ylabel('Количество')
axes[0].set_title('Распределение индекса уязвимости')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

components_for_plot = components_normalized.copy()
components_for_plot.columns = component_names
components_for_plot.boxplot(ax=axes[1])
axes[1].set_ylabel('Нормализованное значение')
axes[1].set_title('Распределение компонентов')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Квартили

df['vulnerability_quartile'] = pd.qcut(df['vulnerability_index_norm'], 
                                           q=4, 
                                           labels=['Q1 (наименее уязвимые)', 'Q2', 'Q3', 'Q4 (наиболее уязвимые)'],
                                           duplicates='drop')


print("\nРаспределение по квартилям:")
print(df['vulnerability_quartile'].value_counts().sort_index())

# ============================================
# ШАГ 5: Анализ связи индекса с негативными исходами
# ============================================

print("\n" + "=" * 80)
print("[5] Анализ связи индекса с негативными исходами")
print("=" * 80)

print("\nДоля негативных исходов по квартилям уязвимости:")
print("=" * 70)

for outcome_var, outcome_name in negative_outcomes:
    print(f"\n{outcome_name}:")
    by_quartile = df.groupby('vulnerability_quartile')[outcome_var].mean() * 100
    
    for quartile, value in by_quartile.items():
        if not pd.isna(value):
            print(f"  {quartile}: {value:.1f}%")
    
    if len(by_quartile) >= 4:
        diff = by_quartile.iloc[-1] - by_quartile.iloc[0]
        print(f"  Разница Q4 - Q1: {diff:.1f}%")

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Связь индекса уязвимости с негативными исходами', fontsize=14)

for idx, (outcome_var, outcome_name) in enumerate(negative_outcomes):
    ax = axes[idx // 2, idx % 2]
    
    by_quartile = df.groupby('vulnerability_quartile')[outcome_var].mean() * 100
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(by_quartile)))
    
    bars = ax.bar(range(len(by_quartile)), by_quartile.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(by_quartile)))
    ax.set_xticklabels(by_quartile.index, rotation=45, ha='right')
    ax.set_ylabel('Доля с негативным исходом (%)')
    ax.set_title(outcome_name)
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars, by_quartile.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================
# ШАГ 6: Метод 2 - Сравнение прогностической силы моделей (AUC-ROC)
# ============================================

print("\n" + "=" * 80)
print("[6] МЕТОД 2: Сравнение прогностической силы моделей (AUC-ROC)")
print("=" * 80)

# Словарь для хранения результатов по всем исходам
all_outcomes_results = {}

for target_var, target_name in negative_outcomes:
    print(f"\n" + "-" * 40)
    print(f"Целевая переменная: {target_name}")
    print("-" * 40)
    
    # Подготовка данных
    model_a_data = df[[target_var, 'Wealth_Idx_Lb']].dropna()
    model_b_data = df[[target_var, 'Edu_level']].dropna()
    model_c_data = df[[target_var, 'vulnerability_index_norm']].dropna()
    
    results = []
    
    def evaluate_model(X, y, name):
        if len(X) < 100 or len(np.unique(y)) < 2:
            return None
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred)
        return {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'model': model, 'name': name}
    
    # Модель A (Wealth)
    X_a = model_a_data[['Wealth_Idx_Lb']].values.reshape(-1, 1)
    y_a = model_a_data[target_var].values
    res_a = evaluate_model(X_a, y_a, f"Wealth")
    if res_a:
        results.append(res_a)
        print(f"Модель Wealth: AUC = {res_a['auc']:.4f}")
    
    # Модель B (Education)
    X_b = model_b_data[['Edu_level']].values.reshape(-1, 1)
    y_b = model_b_data[target_var].values
    res_b = evaluate_model(X_b, y_b, f"Education")
    if res_b:
        results.append(res_b)
        print(f"Модель Education: AUC = {res_b['auc']:.4f}")
    
    # Модель C (Индекс)
    X_c = model_c_data[['vulnerability_index_norm']].values.reshape(-1, 1)
    y_c = model_c_data[target_var].values
    res_c = evaluate_model(X_c, y_c, f"Индекс уязвимости")
    if res_c:
        results.append(res_c)
        print(f"Модель Индекс: AUC = {res_c['auc']:.4f}")
    
    if results:
        all_outcomes_results[target_name] = results
        
        # Сортировка по AUC
        results.sort(key=lambda x: x['auc'], reverse=True)
        print(f"\nЛучшая модель: {results[0]['name']} (AUC={results[0]['auc']:.4f})")

# ============================================
# Визуализация ROC для всех исходов
# ============================================
print("\n--- Визуализация ROC-кривых для всех исходов ---")

n_outcomes = len(all_outcomes_results)
if n_outcomes > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Сравнение ROC-кривых для всех негативных исходов', fontsize=14)
    
    for idx, (target_name, results) in enumerate(all_outcomes_results.items()):
        ax = axes[idx // 2, idx % 2]
        
        colors = ['blue', 'red', 'green']
        for i, res in enumerate(results):
            ax.plot(res['fpr'], res['tpr'], 
                   color=colors[i % len(colors)], 
                   linewidth=2,
                   label=f"{res['name']} (AUC={res['auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(target_name)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================
# ШАГ 7: Метод 3 - Анализ порога индекса
# ============================================

print("\n" + "=" * 80)
print("[7] МЕТОД 3: Анализ порога индекса")
print("=" * 80)

# Децили
try:
    df['vulnerability_decile'] = pd.qcut(df['vulnerability_index_norm'], 
                                          q=10, 
                                          labels=[f'D{i}' for i in range(1, 11)],
                                          duplicates='drop')
    
    print("\nРаспределение по децилям (для первого исхода):")
    first_outcome, first_name = negative_outcomes[0]
    by_decile = df.groupby('vulnerability_decile')[first_outcome].mean() * 100
    
    for decile, value in by_decile.items():
        print(f"  {decile}: {value:.1f}%")
    
    if len(by_decile) >= 10:
        diff = by_decile.iloc[-1] - by_decile.iloc[0]
        print(f"  Разница D10 - D1: {diff:.1f}%")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(by_decile)))
    bars = ax.bar(range(len(by_decile)), by_decile.values, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(by_decile)))
    ax.set_xticklabels(by_decile.index)
    ax.set_xlabel('Дециль индекса уязвимости')
    ax.set_ylabel(f'Доля с {first_name} (%)')
    ax.set_title(f'Градиент риска по децилям')
    ax.set_ylim(0, 100)
    ax.plot(range(len(by_decile)), by_decile.values, 'ro-', linewidth=2, markersize=8, alpha=0.5)
    
    mean_val = by_decile.mean()
    ax.axhline(y=mean_val, color='blue', linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.1f}%')
    
    above_mean = by_decile[by_decile > mean_val]
    if not above_mean.empty:
        first_above = above_mean.index[0]
        ax.axvline(x=list(by_decile.index).index(first_above) - 0.5, 
                  color='purple', linestyle=':', linewidth=2,
                  label=f'Порог: {first_above}')
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Не удалось создать децили: {e}")

# ============================================
# ШАГ 8: Сводная таблица результатов
# ============================================

print("\n" + "=" * 80)
print("[8] Сводная таблица результатов")
print("=" * 80)

summary_data = []

for outcome_var, outcome_name in negative_outcomes:
    overall = df[outcome_var].mean() * 100
    by_quartile = df.groupby('vulnerability_quartile')[outcome_var].mean() * 100
    
    row = {
        'Негативный исход': outcome_name,
        'Общая доля (%)': f"{overall:.1f}%"
    }
    
    for i, (quartile, value) in enumerate(by_quartile.items()):
        if i < 4:
            row[f'Q{i+1}'] = f"{value:.1f}%"
    
    if len(by_quartile) >= 4:
        row['Отношение Q4/Q1'] = f"{by_quartile.iloc[3] / by_quartile.iloc[0]:.2f}"
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\nСводная таблица:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ ПО ГИПОТЕЗЕ 6 ЗАВЕРШЕН")

print("=" * 80)
