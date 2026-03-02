# ============================================
# ГИПОТЕЗА 5: Влияние положения женщины на медицинские практики
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ГИПОТЕЗА 5: Влияние положения женщины на медицинские практики")
print("=" * 80)

# ============================================
# ШАГ 1: Загрузка данных
# ============================================

print("\n[1] Загрузка данных...")
df = pd.read_csv('Final.csv')
print(f"Загружено записей: {len(df)}")

# ============================================
# ШАГ 2: Создание интегрального признака автономии
# ============================================

print("\n[2] Создание интегрального признака автономии женщины...")

# Используем доступные прокси-переменные из датасета
# Образование как показатель автономии
df['woman_decision_power'] = (df['Edu_level'] >= 2).astype(int)

print(f"Доля женщин с высоким образованием (прокси автономии): {df['woman_decision_power'].mean():.2%}")

# ============================================
# ШАГ 3: Создание целевых переменных
# ============================================

print("\n[3] Создание целевых переменных...")

# Роды в медицинском учреждении
df['institutional_delivery'] = (
    (df['DeliveryPlace_Other'] == 1) | 
    (df['DeliveryPlace_Private'] == 1)
).astype(int)
print(f"- Роды в учреждении: {df['institutional_delivery'].mean():.2%}")

# 4+ антенатальных посещений
df['adequate_antenatal'] = (df['Antenatal_visits'] >= 4).astype(int)
print(f"- 4+ антенатальных посещений: {df['adequate_antenatal'].mean():.2%}")

# Полная вакцинация
df['full_vaccination'] = (
    (df['DPT_full'] == 1) & 
    (df['MEASLES_full'] == 1)
).astype(int)
print(f"- Полная вакцинация: {df['full_vaccination'].mean():.2%}")

# Тип поселения
df['urban'] = df['ResidenceType_Urban']
print(f"- Городское население: {df['urban'].mean():.2%}")

# ============================================
# ШАГ 4: Список исходов для анализа
# ============================================

outcomes = [
    ('institutional_delivery', 'Роды в учреждении'),
    ('adequate_antenatal', '4+ антенатальных посещений'),
    ('full_vaccination', 'Полная вакцинация')
]

# ============================================
# ШАГ 5: Тест хи-квадрат (сравнение пропорций)
# ============================================

print("\n" + "=" * 80)
print("[4] Тест хи-квадрат: сравнение пропорций")
print("=" * 80)

for outcome_var, outcome_name in outcomes:
    # Таблица сопряженности
    ct = pd.crosstab(df['woman_decision_power'], df[outcome_var])
    ct.columns = ['Нет', 'Да']
    ct.index = ['Низкое образование', 'Высокое образование']
    
    # Проценты
    pct = pd.crosstab(df['woman_decision_power'], df[outcome_var], normalize='index') * 100
    
    # Хи-квадрат тест
    chi2, p_val, dof, expected = chi2_contingency(ct)
    
    print(f"\n{outcome_name}:")
    print(ct)
    print(f"\nПроценты:")
    print(f"  Низкое образование: {pct.loc[0, 1]:.1f}%")
    print(f"  Высокое образование: {pct.loc[1, 1]:.1f}%")
    print(f"  Разница: {pct.loc[1, 1] - pct.loc[0, 1]:.1f}%")
    print(f"Chi2 = {chi2:.2f}, p-value = {p_val:.4f}")
    print(f"  -> {'Статистически значимо' if p_val < 0.05 else 'Статистически не значимо'}")

# ============================================
# ШАГ 6: Логистическая регрессия с контролем факторов
# ============================================

print("\n" + "=" * 80)
print("[5] Логистическая регрессия (с контролем факторов)")
print("=" * 80)

control_vars = ['Wealth_Idx_Lb']
print(f"Контрольные переменные: {control_vars}")

for outcome_var, outcome_name in outcomes:
    print(f"\n{outcome_name}:")
    
    # Подготовка данных
    model_df = df[['woman_decision_power'] + control_vars + [outcome_var]].dropna()
    
    X = model_df[['woman_decision_power'] + control_vars]
    y = model_df[outcome_var]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Коэффициенты и odds ratios
    for name, coef in zip(X.columns, model.coef_[0]):
        or_val = np.exp(coef)
        print(f"  {name}: coef={coef:.3f}, OR={or_val:.2f}")
    
    # Интерпретация для главной переменной
    idx = list(X.columns).index('woman_decision_power')
    or_main = np.exp(model.coef_[0][idx])
    print(f"\n  Женщины с высоким образованием имеют в {or_main:.2f} раз {'выше' if or_main > 1 else 'ниже'} шанс {outcome_name.lower()}")

# ============================================
# ШАГ 7: Стратификация по городу/селу
# ============================================

print("\n" + "=" * 80)
print("[6] Стратифицированный анализ: город vs село")
print("=" * 80)

for outcome_var, outcome_name in outcomes:
    print(f"\n{outcome_name}:")
    
    # Город
    urban_data = df[df['urban'] == 1]
    urban_pct = pd.crosstab(urban_data['woman_decision_power'], urban_data[outcome_var], normalize='index') * 100
    print(f"\n  ГОРОД (n={len(urban_data)}):")
    print(f"    Низкое образование: {urban_pct.loc[0, 1]:.1f}%")
    print(f"    Высокое образование: {urban_pct.loc[1, 1]:.1f}%")
    
    # Село
    rural_data = df[df['urban'] == 0]
    rural_pct = pd.crosstab(rural_data['woman_decision_power'], rural_data[outcome_var], normalize='index') * 100
    print(f"\n  СЕЛО (n={len(rural_data)}):")
    print(f"    Низкое образование: {rural_pct.loc[0, 1]:.1f}%")
    print(f"    Высокое образование: {rural_pct.loc[1, 1]:.1f}%")

# ============================================
# ШАГ 8: Визуализация
# ============================================

print("\n[7] Построение визуализаций...")

# Создаем 4 группы: город/высокое образование, город/низкое образование, село/высокое образование, село/низкое образование
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Влияние образования женщины на медицинские практики: город vs село', fontsize=14)

for idx, (outcome_var, outcome_name) in enumerate(outcomes):
    ax = axes[idx]
    
    # Рассчитываем проценты для каждой группы
    groups = []
    for urban_val, urban_label in [(1, 'Город'), (0, 'Село')]:
        for edu_val, edu_label in [(1, 'высокое'), (0, 'низкое')]:
            subset = df[(df['urban'] == urban_val) & (df['woman_decision_power'] == edu_val)]
            pct = subset[outcome_var].mean() * 100
            groups.append({
                'label': f"{urban_label}\n{edu_label} обр",
                'value': pct,
                'color': '#2ecc71' if edu_val == 1 else '#e74c3c'
            })
    
    # Построение графика
    x_pos = range(len(groups))
    bars = ax.bar(x_pos, [g['value'] for g in groups], 
                 color=[g['color'] for g in groups], edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g['label'] for g in groups])
    ax.set_ylabel('Процент (%)')
    ax.set_title(outcome_name)
    ax.set_ylim(0, 100)
    
    # Добавляем значения
    for bar, group in zip(bars, groups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{group['value']:.1f}%", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================
# ШАГ 9: Сводная таблица
# ============================================

print("\n" + "=" * 80)
print("[8] Сводная таблица результатов")
print("=" * 80)

summary_data = []
for outcome_var, outcome_name in outcomes:
    row = {'Показатель': outcome_name}
    row['Всего'] = f"{df[outcome_var].mean()*100:.1f}%"
    
    by_edu = df.groupby('woman_decision_power')[outcome_var].mean() * 100
    row['Низкое образование'] = f"{by_edu[0]:.1f}%"
    row['Высокое образование'] = f"{by_edu[1]:.1f}%"
    
    by_urban = df.groupby('urban')[outcome_var].mean() * 100
    row['Село'] = f"{by_urban[0]:.1f}%"
    row['Город'] = f"{by_urban[1]:.1f}%"
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\nСводная таблица:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ ПО ГИПОТЕЗЕ 5 ЗАВЕРШЕН")
print("=" * 80)